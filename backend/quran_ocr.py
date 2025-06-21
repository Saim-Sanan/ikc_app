import cv2
import numpy as np
import pandas as pd
import re
import os
import time
import threading
import pytesseract
import easyocr
from fuzzywuzzy import fuzz, process
from PIL import Image
import arabic_reshaper
from bidi.algorithm import get_display
import logging
import json
from skimage import exposure, filters, transform, segmentation, morphology, feature
from pathlib import Path
from flask import Flask, request, jsonify
import base64
from collections import defaultdict

# Setup logging with cleaner format
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('QuranicExtractor')

# Configuration
DEFAULT_CSV_PATH = 'C:/Users/ABC/OneDrive/Desktop/backend/The-Quran-Dataset.csv'  # Updated default path for Kaggle dataset
OUTPUT_DIR = './extraction_results/'
CACHE_DIR = './cache/'
PREPROCESSING_CACHE = './cache/preprocessed/'

# Ensure directories exist
for dir_path in [OUTPUT_DIR, CACHE_DIR, PREPROCESSING_CACHE]:
    os.makedirs(dir_path, exist_ok=True)

# Set Tesseract path - auto-detect if not found
if not os.path.exists(r'C:/Program Files/Tesseract-OCR/tesseract.exe'):
    # Try common Linux/Mac location
    if os.path.exists('/usr/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'
    else:
        logger.warning("Tesseract not found at default location. Please set the path manually.")
else:
    pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'


class QuranicTextExtractor:
    def __init__(self, csv_path=DEFAULT_CSV_PATH, use_cache=True):
        """
        Initialize with path to Quran verses CSV
        
        Args:
            csv_path: Path to CSV file containing Quran verses
            use_cache: Whether to use caching for preprocessing and OCR results
        """
        self.csv_path = csv_path
        self.use_cache = use_cache
        self.easyocr_reader = None  # Lazy loading for easyocr
        self.verse_embeddings = None  # Will be initialized with verse data
        self.verified_data = None
        self.load_verified_data()
        
    def load_verified_data(self):
        """
        Load verified Quran verses from CSV with support for the Kaggle dataset format
        
        The Kaggle dataset has columns:
        surah_no, surah_name_en, surah_name_ar, surah_name_roman, ayah_no_surah, 
        ayah_no_quran, ayah_ar, ayah_en, etc.
        """
        try:
            logger.info(f"Loading Quran verses from {self.csv_path}")
            
            if not os.path.exists(self.csv_path):
                logger.error(f"CSV file not found: {self.csv_path}")
                self.verified_data = {
                    "verses": [], 
                    "surah_nums": [], 
                    "verse_nums": [], 
                    "has_reference": False,
                    "surah_names": [],
                    "verse_translations": [],
                    "global_verse_nums": []
                }
                return
            
            # Try different encoding options in case of Unicode issues
            for encoding in ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-6']:
                try:
                    df = pd.read_csv(self.csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Error with encoding {encoding}: {str(e)}")
                    continue
            else:
                # If all encodings fail
                logger.error("Failed to read CSV with any encoding")
                self.verified_data = {
                    "verses": [], 
                    "surah_nums": [], 
                    "verse_nums": [], 
                    "has_reference": False,
                    "surah_names": [],
                    "verse_translations": [],
                    "global_verse_nums": []
                }
                return
            
            # Check if this is the Kaggle dataset format
            kaggle_format = all(col in df.columns for col in ['surah_no', 'ayah_no_surah', 'ayah_ar'])
            
            if kaggle_format:
                logger.info("Detected Kaggle Quran dataset format")
                
                # Extract relevant columns
                surah_nums = df['surah_no'].astype(str).tolist()
                verse_nums = df['ayah_no_surah'].astype(str).tolist()
                verses = df['ayah_ar'].astype(str).tolist()
                
                # Additional information if available
                surah_names = df['surah_name_en'].astype(str).tolist() if 'surah_name_en' in df.columns else [''] * len(verses)
                surah_names_ar = df['surah_name_ar'].astype(str).tolist() if 'surah_name_ar' in df.columns else [''] * len(verses)
                verse_translations = df['ayah_en'].astype(str).tolist() if 'ayah_en' in df.columns else [''] * len(verses)
                global_verse_nums = df['ayah_no_quran'].astype(str).tolist() if 'ayah_no_quran' in df.columns else [''] * len(verses)
                
                has_reference = True
                logger.info(f"Loaded {len(verses)} verses with references from Kaggle dataset")
                
            # Handle the original format (fallback)
            elif len(df.columns) >= 3:
                # Assuming CSV has surah, verse, text columns
                surah_nums = df.iloc[:, 0].astype(str).tolist()
                verse_nums = df.iloc[:, 1].astype(str).tolist()
                verses = df.iloc[:, 2].astype(str).tolist()
                surah_names = [''] * len(verses)
                surah_names_ar = [''] * len(verses)
                verse_translations = [''] * len(verses)
                global_verse_nums = [''] * len(verses)
                has_reference = True
                logger.info(f"Loaded {len(verses)} verses with references from standard format")
            else:
                # Fallback to single column of verse text
                verses = df.iloc[:, 0].astype(str).tolist()
                surah_nums = ['?'] * len(verses)
                verse_nums = ['?'] * len(verses)
                surah_names = [''] * len(verses)
                surah_names_ar = [''] * len(verses)
                verse_translations = [''] * len(verses)
                global_verse_nums = [''] * len(verses)
                has_reference = False
                logger.info(f"Loaded {len(verses)} verses without references")
            
            # Fix encoding issues in Arabic text if needed
            fixed_verses = []
            for verse in verses:
                # Check if the verse has encoding issues (common with some datasets)
                if not re.search(r'[\u0600-\u06FF]', verse) and re.search(r'[Ø-ÿ]', verse):
                    # This is likely a wrongly encoded Arabic text
                    try:
                        # Try to fix encoding (this is a simplistic approach)
                        verse = verse.encode('latin1').decode('utf-8')
                    except:
                        # If that fails, keep the original
                        pass
                fixed_verses.append(verse)
            
            self.verified_data = {
                "verses": fixed_verses,
                "surah_nums": surah_nums,
                "verse_nums": verse_nums,
                "has_reference": has_reference,
                "surah_names": surah_names,
                "surah_names_ar": surah_names_ar,
                "verse_translations": verse_translations,
                "global_verse_nums": global_verse_nums
            }
            
            # Prepare verse embeddings for faster matching
            self._prepare_verse_embeddings()
            
        except Exception as e:
            logger.error(f"Error loading CSV: {str(e)}")
            self.verified_data = {
                "verses": [], 
                "surah_nums": [], 
                "verse_nums": [], 
                "has_reference": False,
                "surah_names": [],
                "verse_translations": [],
                "global_verse_nums": []
            }
    
    def _prepare_verse_embeddings(self):
        """
        Create simple embeddings for verses to improve matching speed.
        Uses character n-grams as simple features.
        """
        if not self.verified_data or not self.verified_data["verses"]:
            self.verse_embeddings = []
            return
            
        self.verse_embeddings = []
        
        for verse in self.verified_data["verses"]:
            normalized = self.normalize_arabic_text(verse)
            # Create character 3-grams
            ngrams = set()
            for i in range(len(normalized) - 2):
                ngrams.add(normalized[i:i+3])
            self.verse_embeddings.append(ngrams)
    
    def get_cache_path(self, image_path, stage="preprocessing"):
        """Generate cache path for the given image and processing stage"""
        if not image_path:
            return None
            
        image_name = os.path.basename(image_path)
        file_size = os.path.getsize(image_path) if os.path.exists(image_path) else 0
        modification_time = int(os.path.getmtime(image_path)) if os.path.exists(image_path) else 0
        
        signature = f"{image_name}_{file_size}_{modification_time}"
        
        if stage == "preprocessing":
            return os.path.join(PREPROCESSING_CACHE, f"{signature}.json")
        elif stage == "ocr":
            return os.path.join(CACHE_DIR, f"{signature}_ocr.json")
        elif stage == "matches":
            return os.path.join(CACHE_DIR, f"{signature}_matches.json")
        return None

    def normalize_arabic_text(self, text):
        """
        Comprehensive Arabic text normalization
        
        - Removes diacritics (tashkeel)
        - Normalizes various Arabic character forms
        - Removes non-Arabic characters
        - Normalizes whitespace
        """
        if not isinstance(text, str):
            return ""
            
        # Remove tatweel (kashida)
        text = re.sub(r'ـ', '', text)
        
        # Remove diacritics (tashkeel)
        diacritics = re.compile("""[\u064B-\u065F\u0670]""")
        text = re.sub(diacritics, '', text)
        
        # Normalize alefs
        text = re.sub(r'[إأآا]', 'ا', text)
        
        # Normalize ya and alif maqsura
        text = re.sub(r'[يى]', 'ي', text)
        
        # Normalize ha
        text = re.sub(r'[ةه]', 'ه', text)
        
        # Normalize waw
        text = re.sub(r'[ؤو]', 'و', text)
        
        # Normalize hamza
        text = re.sub(r'[ءئؤإأ]', 'ء', text)

        # Remove any non-Arabic characters except spaces
        text = re.sub(r'[^\u0600-\u06FF\s]', '', text)
        
        # Remove extra spaces and normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def detect_text_regions(self, image):
        """
        Detect potential text regions in an image using MSER feature detection
        and morphological operations. This helps isolate text areas for better OCR.
        """
        # Ensure image is grayscale
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Apply adaptive thresholding to create binary image
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)
        
        # Use MSER to detect text regions (optimized parameters)
        mser = cv2.MSER_create(min_area=100, max_area=8000)
        regions, _ = mser.detectRegions(gray)
        
        # Create mask for text regions
        mask = np.zeros(gray.shape, dtype=np.uint8)
        for region in regions:
            for x, y in region:
                mask[y, x] = 255
                
        # Dilate to connect nearby regions
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Close gaps within text regions
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours to identify connected regions
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:  # Filter out small noise
                cv2.drawContours(mask, [contour], 0, 0, -1)
                
        return mask

    def preprocess_image(self, image_path):
        """
        Advanced Image Preprocessing for Arabic text recognition
        """
        try:
            # Check cache first if enabled
            if self.use_cache:
                cache_path = self.get_cache_path(image_path, "preprocessing")
                if cache_path and os.path.exists(cache_path):
                    try:
                        with open(cache_path, 'r') as f:
                            cache_data = json.load(f)
                            logger.info(f"Using cached preprocessing for {image_path}")
                            return [(name, path) for name, path in cache_data if os.path.exists(path)]
                    except Exception as e:
                        logger.warning(f"Cache read error: {str(e)}")
                        # Continue with preprocessing if cache read fails
            
            logger.info(f"Preprocessing image: {image_path}")
            original = cv2.imread(image_path)
            if original is None:
                raise FileNotFoundError(f"Image not found or could not be read: {image_path}")

            # Store all processed versions
            processed_images = []
            img_name = Path(image_path).stem
            
            # Basic grayscale
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            gray_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_gray.jpg")
            cv2.imwrite(gray_path, gray)
            processed_images.append(("Gray", gray_path))
            
            # Detect text regions
            text_mask = self.detect_text_regions(gray)
            masked_text = cv2.bitwise_and(gray, gray, mask=text_mask)
            masked_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_masked.jpg")
            cv2.imwrite(masked_path, masked_text)
            processed_images.append(("Masked Text", masked_path))
            
            # 1. Adaptive thresholding (good for varying backgrounds)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
            thresh_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_adaptive.jpg")
            cv2.imwrite(thresh_path, thresh)
            processed_images.append(("Adaptive Threshold", thresh_path))
            
            # 2. Inverted adaptive thresholding (for light text on dark background)
            thresh_inv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY_INV, 11, 2)
            thresh_inv_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_adaptive_inv.jpg")
            cv2.imwrite(thresh_inv_path, thresh_inv)
            processed_images.append(("Inverted Adaptive", thresh_inv_path))
            
            # 3. Local contrast enhancement 
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(gray)
            enhanced_thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
            enhanced_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_enhanced.jpg")
            cv2.imwrite(enhanced_path, enhanced_thresh)
            processed_images.append(("Enhanced", enhanced_path))
            
            # 4. Noise reduction + thresholding
            denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            _, denoised_thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            denoised_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_denoised.jpg")
            cv2.imwrite(denoised_path, denoised_thresh)
            processed_images.append(("Denoised", denoised_path))
            
            # 5. Edge enhancement
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(gray, -1, kernel)
            _, sharpened_thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            sharpened_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_sharpened.jpg")
            cv2.imwrite(sharpened_path, sharpened_thresh)
            processed_images.append(("Sharpened", sharpened_path))
            
            # 6. Binary thresholds at different levels
            for threshold in [120, 140]:
                _, binary = cv2.threshold(enhanced, threshold, 255, cv2.THRESH_BINARY)
                binary_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_bin_{threshold}.jpg")
                cv2.imwrite(binary_path, binary)
                processed_images.append((f"Binary_{threshold}", binary_path))
            
            # 7. Morphological operations on inverted threshold
            kernel = np.ones((2, 2), np.uint8)
            morph = cv2.morphologyEx(thresh_inv, cv2.MORPH_CLOSE, kernel)
            morph_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_morph.jpg")
            cv2.imwrite(morph_path, morph)
            processed_images.append(("Morphological", morph_path))
            
            # 8. Skew correction for rotated text
            edges = feature.canny(gray, sigma=1)
            lines = transform.probabilistic_hough_line(edges, line_length=100, line_gap=10)
            
            if lines:
                angles = []
                for line in lines:
                    (x1, y1), (x2, y2) = line
                    if x2 != x1:  # Avoid division by zero
                        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                        # Only consider angles near horizontal
                        if -30 < angle < 30:
                            angles.append(angle)
                
                if angles:
                    # Use median angle to avoid outliers
                    median_angle = np.median(angles)
                    # Correct for horizontal text
                    rotation_angle = 90 - median_angle if median_angle > 45 else -median_angle
                    
                    # Rotate the image to correct skew
                    rotated = transform.rotate(gray, rotation_angle, resize=True, preserve_range=True)
                    rotated = rotated.astype(np.uint8)
                    
                    # Threshold the rotated image
                    _, rotated_binary = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    rotated_path = os.path.join(PREPROCESSING_CACHE, f"{img_name}_rotated.jpg")
                    cv2.imwrite(rotated_path, rotated_binary)
                    processed_images.append(("Deskewed", rotated_path))
            
            # Cache the results if caching is enabled
            if self.use_cache and cache_path:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(processed_images, f)
                except Exception as e:
                    logger.warning(f"Cache write error: {str(e)}")
            
            return processed_images

        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            return []

    def perform_ocr(self, image_path, method="tesseract", lang="ara", config=None):
        """
        Perform OCR with specific configurations for Arabic text
        """
        extracted_text = ""
        
        if method == "tesseract":
            try:
                # Default configuration if none provided
                if config is None:
                    config = '--oem 1 --psm 6'
                
                # For Arabic, ensure proper settings
                if lang == "ara":
                    config += ' --tessdata-dir "C:/Program Files/Tesseract-OCR/tessdata"'
                
                text = pytesseract.image_to_string(image_path, lang=lang, config=config)
                extracted_text = text.strip()
                
                # Apply post-processing for Arabic
                if lang == "ara":
                    # Remove non-Arabic characters but keep some punctuation
                    extracted_text = re.sub(r'[^\u0600-\u06FF\s،.:؟!]', '', extracted_text)
                    # Normalize multiple spaces
                    extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
                    
            except Exception as e:
                logger.error(f"Tesseract OCR error: {e}")
                
        elif method == "easyocr":
            try:
                # Lazy loading of EasyOCR reader
                if self.easyocr_reader is None:
                    self.easyocr_reader = easyocr.Reader(['ar'])
                    
                # Use EasyOCR with Arabic language
                result = self.easyocr_reader.readtext(image_path, detail=0)
                extracted_text = " ".join(result)
                
                # Post-processing
                extracted_text = re.sub(r'[^\u0600-\u06FF\s،.:؟!]', '', extracted_text)
                extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
                
            except Exception as e:
                logger.error(f"EasyOCR error: {e}")
        
        return extracted_text

    def extract_text_multi_method(self, image_paths):
        """
        Extract text using multiple OCR methods and preprocessing techniques
        """
        # Check cache if enabled
        if self.use_cache and image_paths:
            original_image = image_paths[0][1]  # Use first image to get original name
            cache_path = self.get_cache_path(original_image, "ocr")
            if cache_path and os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_results = json.load(f)
                        logger.info(f"Using cached OCR results for {original_image}")
                        return cached_results
                except Exception as e:
                    logger.warning(f"OCR cache read error: {str(e)}")
        
        all_texts = []
        # Optimized configs for Arabic text
        tesseract_configs = [
            ('--oem 1 --psm 6', 'Single Block'),  # Single uniform block of text
            ('--oem 1 --psm 4', 'Single Column'),  # Single column of text
            ('--oem 1 --psm 3', 'Auto Page'),      # Automatic page segmentation
        ]
        
        # Process each preprocessed image
        for name, path in image_paths:
            # Process with Tesseract
            for config, config_name in tesseract_configs:
                tesseract_text = self.perform_ocr(path, method="tesseract", config=config)
                source = f"Tesseract-{name}-{config_name}"
                if tesseract_text:
                    all_texts.append((source, tesseract_text))
            
            # Try EasyOCR as well (may work better for some Arabic texts)
            try:
                easyocr_text = self.perform_ocr(path, method="easyocr")
                if easyocr_text:
                    all_texts.append((f"EasyOCR-{name}", easyocr_text))
            except Exception as e:
                logger.warning(f"EasyOCR failed on {name}: {str(e)}")
        
        # Filter texts with too few Arabic characters (likely noise)
        filtered_texts = []
        for source, text in all_texts:
            # Count Arabic characters
            arabic_char_count = len(re.findall(r'[\u0600-\u06FF]', text))
            
            # Require a minimum number of Arabic characters
            min_chars = 5  # Adjust based on shortest verse
            if arabic_char_count >= min_chars:
                filtered_texts.append((source, text, arabic_char_count))
        
        # Sort by number of Arabic characters (more is likely better)
        filtered_texts.sort(key=lambda x: x[2], reverse=True)
        
        # Log results
        logger.info(f"Extracted {len(filtered_texts)} valid text candidates from {len(image_paths)} images")
        
        # Cache results if enabled
        if self.use_cache and image_paths and filtered_texts:
            original_image = image_paths[0][1]
            cache_path = self.get_cache_path(original_image, "ocr")
            if cache_path:
                try:
                    with open(cache_path, 'w', encoding='utf-8') as f:
                        json.dump(filtered_texts, f, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"OCR cache write error: {str(e)}")
        
        return filtered_texts

    def filter_candidates(self, extracted_texts, min_chars=10):
        """
        Filter and cluster similar OCR results to get unique candidates
        """
        if not extracted_texts:
            return []
        
        # Filter by minimum character count
        candidates = [t for t in extracted_texts if t[2] >= min_chars]
        if not candidates:
            return []
        
        # Normalize all texts
        normalized_texts = [(src, self.normalize_arabic_text(text), count) for src, text, count in candidates]
        
        # Cluster similar texts using fuzzy matching
        clusters = []
        for src, norm_text, count in normalized_texts:
            # Check if this text is similar to any existing cluster
            found_cluster = False
            for i, cluster in enumerate(clusters):
                cluster_norm_text = cluster[0][1]
                similarity = fuzz.ratio(norm_text, cluster_norm_text)
                
                # If similar enough, add to cluster
                if similarity > 80:  # 80% similarity threshold
                    clusters[i].append((src, norm_text, count))
                    found_cluster = True
                    break
            
            # If not similar to any cluster, create new cluster
            if not found_cluster:
                clusters.append([(src, norm_text, count)])
        
        # From each cluster, select the text with the most Arabic characters
        unique_candidates = []
        for cluster in clusters:
            best = max(cluster, key=lambda x: x[2])
            
            # Find the original non-normalized version
            for src, text, count in candidates:
                if src == best[0] and count == best[2]:
                    unique_candidates.append((src, text, count))
                    break
        
        return unique_candidates

    def match_verses(self, extracted_texts, min_similarity=60):
        """
        Match extracted text with verified Quran verses using advanced algorithms
        """
        # Check cache if enabled
        if self.use_cache and extracted_texts:
            source_text = extracted_texts[0][1]  # Use first text for cache key
            cache_key = f"{hash(source_text)}_{min_similarity}"
            cache_path = os.path.join(CACHE_DIR, f"matches_{cache_key}.json")
            
            if os.path.exists(cache_path):
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_matches = json.load(f)
                        logger.info(f"Using cached matches")
                        return cached_matches
                except Exception as e:
                    logger.warning(f"Match cache read error: {str(e)}")
        
        if not extracted_texts or not self.verified_data or not self.verified_data["verses"]:
            return []
            
        # Filter and cluster similar OCR results
        unique_candidates = self.filter_candidates(extracted_texts)
        if not unique_candidates:
            return []
        
        all_matches = []
        verses = self.verified_data["verses"]
        surah_nums = self.verified_data["surah_nums"]
        verse_nums = self.verified_data["verse_nums"]
        has_reference = self.verified_data["has_reference"]
        surah_names = self.verified_data["surah_names"]
        surah_names_ar = self.verified_data["surah_names_ar"]
        verse_translations = self.verified_data["verse_translations"]
        global_verse_nums = self.verified_data["global_verse_nums"]
        
        # Prepare normalized versions of verified verses once
        normalized_verified = [self.normalize_arabic_text(verse) for verse in verses]
        
        for source, text, _ in unique_candidates:
            normalized_extracted = self.normalize_arabic_text(text)
            
            # Skip very short texts
            if len(normalized_extracted) < 10:
                continue
                
            # Fast filtering using n-gram embeddings
            potential_matches = []
            extracted_ngrams = set()
            for i in range(len(normalized_extracted) - 2):
                if i + 3 <= len(normalized_extracted):
                    extracted_ngrams.add(normalized_extracted[i:i+3])
            
            for idx, verse_ngrams in enumerate(self.verse_embeddings):
                # Skip empty matches
                if not verse_ngrams:
                    continue
                    
                common_ngrams = extracted_ngrams.intersection(verse_ngrams)
                similarity = len(common_ngrams) / max(len(extracted_ngrams), len(verse_ngrams))
                if similarity > 0.2:  # Threshold for potential matches
                    potential_matches.append((idx, similarity))
            
            # Sort potential matches by similarity
            potential_matches.sort(key=lambda x: x[1], reverse=True)
            
            # Take top 20 candidates for more detailed matching
            potential_matches = potential_matches[:20]
            
            # Refine matches using fuzzy matching
            for idx, _ in potential_matches:
                verified_text = normalized_verified[idx]
                similarity = fuzz.ratio(normalized_extracted, verified_text)
                
                if similarity >= min_similarity:
                    match_info = {
                        "source": source,
                        "extracted_text": text,
                        "verified_text": verses[idx],
                        "similarity": similarity,
                        "surah_num": surah_nums[idx] if has_reference else None,
                        "verse_num": verse_nums[idx] if has_reference else None,
                        "surah_name": surah_names[idx] if has_reference else None,
                        "surah_name_ar": surah_names_ar[idx] if has_reference else None,
                        "verse_translation": verse_translations[idx] if has_reference else None,
                        "global_verse_num": global_verse_nums[idx] if has_reference else None
                    }
                    all_matches.append(match_info)
            
            # If no matches found, try partial matching with lower similarity threshold
            if not all_matches:
                for idx, verse in enumerate(normalized_verified):
                    # Check if the extracted text is a substring of any verse
                    if len(normalized_extracted) >= 15 and normalized_extracted in verse:
                        match_info = {
                            "source": source,
                            "extracted_text": text,
                            "verified_text": verses[idx],
                            "similarity": 50,  # Lower confidence for substring match
                            "surah_num": surah_nums[idx] if has_reference else None,
                            "verse_num": verse_nums[idx] if has_reference else None,
                            "surah_name": surah_names[idx] if has_reference else None,
                            "surah_name_ar": surah_names_ar[idx] if has_reference else None,
                            "verse_translation": verse_translations[idx] if has_reference else None,
                            "global_verse_num": global_verse_nums[idx] if has_reference else None,
                            "match_type": "substring"
                        }
                        all_matches.append(match_info)
        
        # Sort matches by similarity
        all_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Cache results if enabled
        if self.use_cache and extracted_texts and all_matches:
            source_text = extracted_texts[0][1]
            cache_key = f"{hash(source_text)}_{min_similarity}"
            cache_path = os.path.join(CACHE_DIR, f"matches_{cache_key}.json")
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(all_matches, f, ensure_ascii=False)
            except Exception as e:
                logger.warning(f"Match cache write error: {str(e)}")
        
        return all_matches

    def group_matches_by_verse(self, matches):
        """
        Group matches by verse to consolidate duplicate results
        
        Args:
            matches: List of match dictionaries
            
        Returns:
            List of consolidated match dictionaries
        """
        if not matches:
            return []
            
        # Group matches by surah and verse number
        verse_groups = defaultdict(list)
        
        for match in matches:
            # Create a key based on surah and verse number
            key = f"{match.get('surah_num', '')}-{match.get('verse_num', '')}"
            verse_groups[key].append(match)
        
        # For each group, keep the match with highest similarity
        consolidated_matches = []
        
        for key, group in verse_groups.items():
            # Sort by similarity (highest first)
            group.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Take the best match
            best_match = group[0].copy()
            
            # Add sources from all matches in this group
            sources = [m["source"] for m in group]
            best_match["sources"] = sources
            best_match["source_count"] = len(sources)
            
            # Add all extracted texts
            extracted_texts = [m["extracted_text"] for m in group]
            best_match["all_extracted_texts"] = extracted_texts
            
            consolidated_matches.append(best_match)
        
        # Sort by similarity
        consolidated_matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        return consolidated_matches

    def enrich_candidates_with_verse_info(self, candidates, matches):
        """
        Enrich top candidates with verse information from matches
        
        Args:
            candidates: List of candidate tuples (source, text, arabic_char_count)
            matches: List of match dictionaries
            
        Returns:
            List of enriched candidate dictionaries
        """
        if not candidates or not matches:
            return [{"source": source, "text": text, "arabic_chars": count} 
                    for source, text, count in candidates[:3]]
        
        # Create a mapping of extracted text to match info
        text_to_match = {}
        for match in matches:
            text_to_match[match["extracted_text"]] = match
        
        # Enrich candidates with verse info if available
        enriched_candidates = []
        
        for source, text, count in candidates:
            candidate_info = {
                "source": source,
                "text": text,
                "arabic_chars": count
            }
            
            # If this text has a match, add the verse info
            if text in text_to_match:
                match = text_to_match[text]
                candidate_info.update({
                    "surah_num": match.get("surah_num"),
                    "verse_num": match.get("verse_num"),
                    "surah_name": match.get("surah_name"),
                    "surah_name_ar": match.get("surah_name_ar"),
                    "verse_translation": match.get("verse_translation"),
                    "global_verse_num": match.get("global_verse_num"),
                    "similarity": match.get("similarity"),
                    "verified_text": match.get("verified_text")
                })
            
            enriched_candidates.append(candidate_info)
        
        return enriched_candidates[:3]  # Return top 3

    def process_image(self, image_path):
        """
        Process an image to extract and verify Quranic text
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with results
        """
        start_time = time.time()
        
        # Check if file exists
        if not os.path.exists(image_path):
            return {
                "success": False,
                "error": f"Image not found: {image_path}"
            }
        
        # Step 1: Preprocess the image
        preprocessed_images = self.preprocess_image(image_path)
        if not preprocessed_images:
            return {
                "success": False,
                "error": "Failed to preprocess image"
            }
        
        # Step 2: Extract text from preprocessed images
        extracted_texts = self.extract_text_multi_method(preprocessed_images)
        if not extracted_texts:
            return {
                "success": False,
                "error": "No Arabic text detected in image"
            }
        
        # Step 3: Match extracted text with verified verses
        matches = self.match_verses(extracted_texts)
        
        # Step 4: Group matches by verse to consolidate duplicates
        consolidated_matches = self.group_matches_by_verse(matches)
        
        # Step 5: Enrich top candidates with verse information
        enriched_candidates = self.enrich_candidates_with_verse_info(extracted_texts, matches)
        
        processing_time = time.time() - start_time
        
        # Prepare the result
        result = {
            "success": True,
            "processing_time": f"{processing_time:.2f} seconds",
            "matches": consolidated_matches,
            "match_count": len(consolidated_matches),
            "top_candidates": enriched_candidates
        }
        
        return result


# Function for processing through command line
def process_image_cli():
    """Command line function for processing images"""
    print("Quranic Text Extractor - Command Line Interface")
    print("-" * 50)
    
    # Use default CSV path or get from user
    csv_path = input("Enter path to Quran verses CSV file (press Enter for default): ")
    if not csv_path:
        csv_path = DEFAULT_CSV_PATH
    
    # Initialize extractor
    extractor = QuranicTextExtractor(csv_path=csv_path)
    
    # Get image path
    image_path = input("Enter path to image file: ")
    if not image_path or not os.path.exists(image_path):
        print(f"Error: Image not found - {image_path}")
        return
    
    print(f"\nProcessing image: {image_path}")
    print("This may take a moment...\n")
    
    # Process the image
    result = extractor.process_image(image_path)
    
    # Display results
    if not result["success"]:
        print(f"Error: {result['error']}")
        return
    
    print(f"Processing completed in {result['processing_time']}")
    print(f"Found {result['match_count']} unique verse matches\n")
    
    # Display matches
    if result["matches"]:
        print("Top matches:")
        print("-" * 50)
        for i, match in enumerate(result["matches"][:5]):  # Show top 5 matches
            print(f"Match #{i+1} (Similarity: {match['similarity']}%)")
            print(f"Sources: {', '.join(match['sources'][:3])}... ({match['source_count']} total)")
            print(f"Extracted: {match['extracted_text']}")
            print(f"Verified: {match['verified_text']}")
            
            # Display reference information if available
            if match['surah_num'] and match['verse_num']:
                print(f"Reference: Surah {match['surah_num']} ({match['surah_name']}), Verse {match['verse_num']}")
                if match['verse_translation']:
                    print(f"Translation: {match['verse_translation']}")
            
            print("-" * 50)
    else:
        print("No matches found in the Quran verses database")


# Flask API function
def create_flask_app():
    """Create and configure Flask API for mobile app integration"""
    app = Flask(__name__)
    
    # Initialize extractor with default CSV
    extractor = QuranicTextExtractor(csv_path=DEFAULT_CSV_PATH)
    
    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint"""
        return jsonify({
            "status": "healthy",
            "verses_loaded": len(extractor.verified_data["verses"]) if extractor.verified_data else 0,
            "dataset_type": "Kaggle Quran dataset" if extractor.verified_data and extractor.verified_data["has_reference"] else "Unknown"
        })
    
    @app.route('/process', methods=['POST'])
    def process_image():
        """Process image endpoint"""
        try:
            # Get image from request
            if 'image' not in request.files and 'image_base64' not in request.json:
                return jsonify({"success": False, "error": "No image provided"}), 400
            
            # Handle file upload
            if 'image' in request.files:
                image_file = request.files['image']
                
                # Save temporarily
                temp_path = os.path.join(CACHE_DIR, f"temp_{int(time.time())}.jpg")
                image_file.save(temp_path)
                
            # Handle base64 image
            elif 'image_base64' in request.json:
                image_data = request.json['image_base64']
                
                # Detect if the data is a URL or base64
                if image_data.startswith('http'):
                    return jsonify({"success": False, "error": "URL images not supported, send base64 encoded image"}), 400
                
                # Remove header if present (e.g., data:image/jpeg;base64,)
                if ';base64,' in image_data:
                    image_data = image_data.split(';base64,')[1]
                
                # Decode and save
                temp_path = os.path.join(CACHE_DIR, f"temp_{int(time.time())}.jpg")
                with open(temp_path, 'wb') as f:
                    f.write(base64.b64decode(image_data))
            
            # Process the image
            result = extractor.process_image(temp_path)
            
            # Clean up
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            
            return jsonify(result)
                
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    @app.route('/load-csv', methods=['POST'])
    def load_custom_csv():
        """Load custom CSV file"""
        try:
            if 'csv_file' not in request.files:
                return jsonify({"success": False, "error": "No CSV file provided"}), 400
                
            csv_file = request.files['csv_file']
            temp_path = os.path.join(CACHE_DIR, f"custom_csv_{int(time.time())}.csv")
            csv_file.save(temp_path)
            
            # Re-initialize extractor with new CSV
            extractor.csv_path = temp_path
            extractor.load_verified_data()
            
            return jsonify({
                "success": True, 
                "verses_loaded": len(extractor.verified_data["verses"]) if extractor.verified_data else 0,
                "has_references": extractor.verified_data["has_reference"] if extractor.verified_data else False
            })
            
        except Exception as e:
            logger.error(f"API error: {str(e)}")
            return jsonify({"success": False, "error": str(e)}), 500
    
    return app


# Main function - uncomment the one you want to use
if __name__ == "__main__":
    # For command line usage
    # process_image_cli()
    
    # For Flask API usage - uncomment to use
    app = create_flask_app()
    app.run(host='0.0.0.0', port=5000, debug=False)