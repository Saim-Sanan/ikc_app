import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:ikc_app/modules/quran_ocr/controllers/ocr_controller.dart';

class LoadingAyahWidget extends StatelessWidget {
  const LoadingAyahWidget({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = Get.find<OcrController>();

    return Scaffold(
      body: Container(
        width: double.infinity,
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Theme.of(context).primaryColor,
              Theme.of(context).primaryColor.withOpacity(0.8),
            ],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // Processing indicator
                const CircularProgressIndicator(
                  valueColor: AlwaysStoppedAnimation<Color>(Colors.white),
                  strokeWidth: 3,
                ),
                const SizedBox(height: 30),

                // Processing text
                const Text(
                  'Processing Image',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 10),

                // Processing description
                const Text(
                  'Please wait while we analyze the Quranic text in your image',
                  textAlign: TextAlign.center,
                  style: TextStyle(fontSize: 16, color: Colors.white70),
                ),
                const SizedBox(height: 50),

                // Random Ayah display
                Obx(() {
                  if (controller.randomAyahs.isEmpty) {
                    return const SizedBox.shrink();
                  }

                  final currentAyah =
                      controller.randomAyahs[controller.currentAyahIndex.value];

                  return Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(15),
                    ),
                    child: Padding(
                      padding: const EdgeInsets.all(20.0),
                      child: Column(
                        children: [
                          // Arabic text
                          Text(
                            currentAyah.text,
                            style: const TextStyle(fontSize: 22, height: 1.5),
                            textAlign: TextAlign.center,
                            textDirection: TextDirection.rtl,
                          ),
                          const SizedBox(height: 20),

                          // Translation
                          Text(
                            currentAyah.translation,
                            style: const TextStyle(
                              fontSize: 16,
                              fontStyle: FontStyle.italic,
                              height: 1.4,
                            ),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 15),

                          // Reference
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 15,
                              vertical: 5,
                            ),
                            decoration: BoxDecoration(
                              color: Theme.of(
                                context,
                              ).primaryColor.withOpacity(0.1),
                              borderRadius: BorderRadius.circular(20),
                            ),
                            child: Text(
                              'Surah ${currentAyah.surah} (${currentAyah.numberInSurah})',
                              style: TextStyle(
                                fontWeight: FontWeight.bold,
                                color: Theme.of(context).primaryColor,
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  );
                }),

                const Spacer(),

                // Cancel button
                TextButton.icon(
                  onPressed: () {
                    controller.isProcessing.value = false;
                    Get.back();
                  },
                  icon: const Icon(Icons.cancel, color: Colors.white70),
                  label: const Text(
                    'Cancel',
                    style: TextStyle(color: Colors.white70),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
