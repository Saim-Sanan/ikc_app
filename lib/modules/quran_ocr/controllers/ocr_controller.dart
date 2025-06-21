import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:get/get.dart';
import 'package:ikc_app/modules/quran_ocr/models/ocr_result.dart';
import 'package:ikc_app/modules/quran_ocr/models/random_ayah.dart';
import 'package:ikc_app/modules/quran_ocr/services/api_service.dart';
import 'package:image_picker/image_picker.dart';
import 'package:just_audio/just_audio.dart';

class OcrController extends GetxController {
  final ApiService _apiService = ApiService();
  final ImagePicker _imagePicker = ImagePicker();
  final AudioPlayer audioPlayer = AudioPlayer();

  // Observable variables
  final Rx<File?> selectedImage = Rx<File?>(null);
  final RxBool isProcessing = false.obs;
  final Rx<OcrResult?> ocrResult = Rx<OcrResult?>(null);
  final RxString errorMessage = ''.obs;
  final RxList<RandomAyah> randomAyahs = <RandomAyah>[].obs;
  final RxInt currentAyahIndex = 0.obs;

  // Audio playback state
  final RxString currentlyPlayingAyah = ''.obs;
  final RxBool isAudioLoading = false.obs;
  final RxBool isAudioPlaying = false.obs;

  // Timer for rotating random ayahs
  RxInt ayahRotationSeconds = 0.obs;

  @override
  void onInit() {
    super.onInit();
    // Load initial random ayah
    loadRandomAyah();

    // Listen to audio player state changes
    audioPlayer.playerStateStream.listen((state) {
      if (state.processingState == ProcessingState.completed) {
        isAudioPlaying.value = false;
        currentlyPlayingAyah.value = '';
      }
    });
  }

  @override
  void onClose() {
    audioPlayer.dispose();
    super.onClose();
  }

  // Pick image from camera
  Future<void> pickImageFromCamera() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.camera,
      imageQuality: 80,
    );

    if (image != null) {
      selectedImage.value = File(image.path);
      ocrResult.value = null;
      errorMessage.value = '';
    }
  }

  // Pick image from gallery
  Future<void> pickImageFromGallery() async {
    final XFile? image = await _imagePicker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 80,
    );

    if (image != null) {
      selectedImage.value = File(image.path);
      ocrResult.value = null;
      errorMessage.value = '';
    }
  }

  // Process selected image
  Future<void> processImage() async {
    if (selectedImage.value == null) {
      errorMessage.value = 'Please select an image first';
      return;
    }

    try {
      isProcessing.value = true;
      errorMessage.value = '';

      // Start loading random ayahs for display during processing
      startAyahRotation();

      // Process the image
      final result = await _apiService.processImage(selectedImage.value!);

      if (result.success) {
        ocrResult.value = result;
      } else {
        errorMessage.value = 'Failed to process image';
      }
    } catch (e) {
      errorMessage.value = 'Error: ${e.toString()}';
    } finally {
      isProcessing.value = false;
      // Stop ayah rotation
      ayahRotationSeconds.value = 0;
    }
  }

  // Load a random ayah
  Future<void> loadRandomAyah() async {
    try {
      // Generate random ayah number (1-6236)
      final randomAyahNumber =
          (DateTime.now().millisecondsSinceEpoch % 6236) + 1;

      // Get Arabic version
      final arabicAyah = await _apiService.getRandomAyah(
        edition: 'ar.asad',
        randomAyahNumber: randomAyahNumber,
      );

      // Get English translation
      final englishAyah = await _apiService.getRandomAyah(
        edition: 'en.asad',
        randomAyahNumber: randomAyahNumber,
      );

      // Combine them
      final combinedAyah = RandomAyah(
        number: arabicAyah.number,
        text: arabicAyah.text,
        translation: englishAyah.text,
        surah: arabicAyah.surah,
        numberInSurah: arabicAyah.numberInSurah,
      );

      randomAyahs.add(combinedAyah);

      // Keep only the last 5 ayahs to avoid memory issues
      if (randomAyahs.length > 5) {
        randomAyahs.removeAt(0);
      }

      // Update current index to show the newest ayah
      currentAyahIndex.value = randomAyahs.length - 1;
    } catch (e) {
      if (kDebugMode) {
        print('Error loading random ayah: $e');
      }
    }
  }

  // Start rotating random ayahs during processing
  void startAyahRotation() async {
    // Reset counter
    ayahRotationSeconds.value = 0;

    // Load initial ayah if empty
    if (randomAyahs.isEmpty) {
      await loadRandomAyah();
    }

    // Start rotation timer
    while (isProcessing.value) {
      await Future.delayed(const Duration(seconds: 1));
      ayahRotationSeconds.value++;

      // Load a new ayah every 10 seconds
      if (ayahRotationSeconds.value % 10 == 0) {
        await loadRandomAyah();
      }

      // Rotate through available ayahs every 5 seconds
      if (ayahRotationSeconds.value % 5 == 0 && randomAyahs.length > 1) {
        currentAyahIndex.value =
            (currentAyahIndex.value + 1) % randomAyahs.length;
      }
    }
  }

  // Play ayah audio
  Future<void> playAyahAudio(String globalVerseNum) async {
    // Stop any currently playing audio
    await audioPlayer.stop();

    // Reset states
    isAudioPlaying.value = false;
    isAudioLoading.value = true;
    currentlyPlayingAyah.value = globalVerseNum;

    try {
      // Get audio URL
      final audioUrl = _apiService.getAyahAudioUrl(globalVerseNum);

      // Load and play audio
      await audioPlayer.setUrl(audioUrl);
      await audioPlayer.play();
      isAudioPlaying.value = true;
    } catch (e) {
      if (kDebugMode) {
        print('Error playing audio: $e');
      }
      currentlyPlayingAyah.value = '';
    } finally {
      isAudioLoading.value = false;
    }
  }

  // Stop audio playback
  Future<void> stopAudio() async {
    await audioPlayer.stop();
    isAudioPlaying.value = false;
    currentlyPlayingAyah.value = '';
  }

  // Reset everything
  void reset() {
    selectedImage.value = null;
    ocrResult.value = null;
    errorMessage.value = '';
    stopAudio();
  }
}
