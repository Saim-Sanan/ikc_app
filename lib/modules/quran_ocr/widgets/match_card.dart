import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:ikc_app/modules/quran_ocr/controllers/ocr_controller.dart';
import 'package:ikc_app/modules/quran_ocr/models/quran_match.dart';

class MatchCard extends StatelessWidget {
  final QuranMatch match;

  const MatchCard({super.key, required this.match});

  @override
  Widget build(BuildContext context) {
    final controller = Get.find<OcrController>();

    return Card(
      margin: const EdgeInsets.only(bottom: 16),
      elevation: 2,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(
          color: _getSimilarityColor(match.similarity),
          width: 2,
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Similarity indicator and Surah info
            Row(
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 5,
                  ),
                  decoration: BoxDecoration(
                    color: _getSimilarityColor(
                      match.similarity,
                    ).withOpacity(0.2),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(
                        Icons.check_circle,
                        size: 16,
                        color: _getSimilarityColor(match.similarity),
                      ),
                      const SizedBox(width: 5),
                      Text(
                        '${match.similarity}% Match',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: _getSimilarityColor(match.similarity),
                        ),
                      ),
                    ],
                  ),
                ),
                const Spacer(),
                Text(
                  'Surah ${match.surahNum}, Ayah ${match.verseNum}',
                  style: const TextStyle(fontWeight: FontWeight.bold),
                ),
              ],
            ),
            const SizedBox(height: 15),

            // Surah name and audio button
            if (match.surahName.isNotEmpty && match.surahNameAr.isNotEmpty)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.symmetric(
                  vertical: 8,
                  horizontal: 12,
                ),
                decoration: BoxDecoration(
                  color: Theme.of(context).primaryColor.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    // Surah name
                    Expanded(
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Text(
                            match.surahName,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 16,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Text(
                            match.surahNameAr,
                            style: const TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 18,
                            ),
                            textDirection: TextDirection.rtl,
                          ),
                        ],
                      ),
                    ),

                    // Audio button
                    if (match.globalVerseNum.isNotEmpty)
                      Obx(() {
                        final isPlaying =
                            controller.isAudioPlaying.value &&
                            controller.currentlyPlayingAyah.value ==
                                match.globalVerseNum;
                        final isLoading =
                            controller.isAudioLoading.value &&
                            controller.currentlyPlayingAyah.value ==
                                match.globalVerseNum;

                        return IconButton(
                          onPressed:
                              isLoading
                                  ? null
                                  : () {
                                    if (isPlaying) {
                                      controller.stopAudio();
                                    } else {
                                      controller.playAyahAudio(
                                        match.globalVerseNum,
                                      );
                                    }
                                  },
                          icon:
                              isLoading
                                  ? const SizedBox(
                                    width: 20,
                                    height: 20,
                                    child: CircularProgressIndicator(
                                      strokeWidth: 2,
                                    ),
                                  )
                                  : Icon(
                                    isPlaying
                                        ? Icons.pause_circle_filled
                                        : Icons.play_circle_filled,
                                    color: Theme.of(context).primaryColor,
                                    size: 30,
                                  ),
                          tooltip: isPlaying ? 'Stop Audio' : 'Play Audio',
                        );
                      }),
                  ],
                ),
              ),
            const SizedBox(height: 15),

            // Verified text (Arabic)
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey[100],
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                match.verifiedText,
                style: const TextStyle(fontSize: 20, height: 1.5),
                textAlign: TextAlign.right,
                textDirection: TextDirection.rtl,
              ),
            ),
            const SizedBox(height: 15),

            // Translation
            if (match.verseTranslation.isNotEmpty)
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.grey[50],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey[300]!),
                ),
                child: Text(
                  match.verseTranslation,
                  style: const TextStyle(
                    fontSize: 16,
                    fontStyle: FontStyle.italic,
                    height: 1.4,
                  ),
                ),
              ),
            const SizedBox(height: 15),

            // Extracted text section
            const Text(
              'Extracted Text:',
              style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
            ),
            const SizedBox(height: 5),
            Container(
              width: double.infinity,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.grey[200],
                borderRadius: BorderRadius.circular(8),
              ),
              child: Text(
                match.extractedText,
                style: const TextStyle(fontSize: 16, height: 1.5),
                textAlign: TextAlign.right,
                textDirection: TextDirection.rtl,
              ),
            ),
            const SizedBox(height: 10),

            // Source info
            Row(
              children: [
                const Icon(Icons.info_outline, size: 16, color: Colors.grey),
                const SizedBox(width: 5),
                Expanded(
                  child: Text(
                    'Source: ${match.source} (${match.sourceCount} sources)',
                    style: const TextStyle(color: Colors.grey, fontSize: 12),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Color _getSimilarityColor(int similarity) {
    if (similarity >= 80) return Colors.green;
    if (similarity >= 70) return Colors.lightGreen;
    if (similarity >= 60) return Colors.amber;
    return Colors.orange;
  }
}
