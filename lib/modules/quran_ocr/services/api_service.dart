import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;
import 'package:ikc_app/modules/quran_ocr/models/ocr_result.dart';
import 'package:ikc_app/modules/quran_ocr/models/random_ayah.dart';
import 'package:ikc_app/utils/constants.dart';

class ApiService {
  // Process image with OCR API
  Future<OcrResult> processImage(File imageFile) async {
    try {
      // Read file as bytes and convert to base64
      final bytes = await imageFile.readAsBytes();
      final base64Image = base64Encode(bytes);

      // Create JSON payload
      final Map<String, dynamic> payload = {'image_base64': base64Image};

      // Send POST request with JSON content type
      final response = await http.post(
        Uri.parse('${ApiConstants.baseUrl}/process'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode(payload),
      );

      // Check response
      if (response.statusCode == 200) {
        final jsonData = jsonDecode(response.body);
        return OcrResult.fromJson(jsonData);
      } else {
        throw Exception(
          'Failed to process image: ${response.statusCode} - ${response.body}',
        );
      }
    } catch (e) {
      return OcrResult.error(e.toString());
    }
  }

  // Get random Quran ayah
  Future<RandomAyah> getRandomAyah({
    String edition = 'ar.asad',
    int randomAyahNumber = 1,
  }) async {
    try {
      final response = await http.get(
        Uri.parse(
          '${ApiConstants.quranApiUrl}/ayah/$randomAyahNumber/$edition',
        ),
      );

      if (response.statusCode == 200) {
        final jsonData = jsonDecode(response.body);
        return RandomAyah.fromJson(jsonData);
      } else {
        throw Exception('Failed to load random ayah');
      }
    } catch (e) {
      // Return a default ayah if API fails
      return RandomAyah(
        number: 1,
        text: 'بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ',
        translation:
            'In the name of Allah, the Entirely Merciful, the Especially Merciful',
        surah: 'Al-Fatihah',
        numberInSurah: 1,
      );
    }
  }

  // Get ayah audio URL
  String getAyahAudioUrl(String globalVerseNum) {
    return 'https://cdn.islamic.network/quran/audio/128/ar.alafasy/$globalVerseNum.mp3';
  }
}
