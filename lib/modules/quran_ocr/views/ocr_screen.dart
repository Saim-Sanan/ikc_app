import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:ikc_app/modules/quran_ocr/controllers/ocr_controller.dart';
import 'package:ikc_app/modules/quran_ocr/views/result_screen.dart';
import 'package:ikc_app/modules/quran_ocr/widgets/custom_button.dart';
import 'package:ikc_app/modules/quran_ocr/widgets/loading_ayah_widget.dart';

class OcrScreen extends StatelessWidget {
  const OcrScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final controller = Get.find<OcrController>();

    return Scaffold(
      appBar: AppBar(title: const Text('Scan Quranic Text')),
      body: Obx(() {
        // Show loading screen while processing
        if (controller.isProcessing.value) {
          return const LoadingAyahWidget();
        }

        // Show results if available
        if (controller.ocrResult.value != null) {
          // Navigate to results screen
          WidgetsBinding.instance.addPostFrameCallback((_) {
            Get.off(() => const ResultScreen());
          });
          return const Center(child: CircularProgressIndicator());
        }

        // Show image selection screen
        return SingleChildScrollView(
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                // Instructions
                Card(
                  elevation: 2,
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      children: [
                        Icon(
                          Icons.info_outline,
                          color: Theme.of(context).primaryColor,
                          size: 30,
                        ),
                        const SizedBox(height: 10),
                        const Text(
                          'Take a clear photo of Quranic text to identify the verse',
                          textAlign: TextAlign.center,
                          style: TextStyle(fontSize: 16),
                        ),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                // Image preview
                Container(
                  height: 300,
                  decoration: BoxDecoration(
                    color: Colors.grey[200],
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: Colors.grey[300]!, width: 1),
                  ),
                  child:
                      controller.selectedImage.value != null
                          ? ClipRRect(
                            borderRadius: BorderRadius.circular(12),
                            child: Image.file(
                              controller.selectedImage.value!,
                              fit: BoxFit.cover,
                            ),
                          )
                          : Center(
                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Icon(
                                  Icons.image_outlined,
                                  size: 80,
                                  color: Colors.grey[400],
                                ),
                                const SizedBox(height: 10),
                                Text(
                                  'No image selected',
                                  style: TextStyle(
                                    color: Colors.grey[600],
                                    fontSize: 16,
                                  ),
                                ),
                              ],
                            ),
                          ),
                ),
                const SizedBox(height: 20),

                // Error message if any
                if (controller.errorMessage.value.isNotEmpty)
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: Colors.red[50],
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.red[200]!),
                    ),
                    child: Text(
                      controller.errorMessage.value,
                      style: TextStyle(color: Colors.red[800]),
                    ),
                  ),

                const SizedBox(height: 20),

                // Image source buttons
                Row(
                  children: [
                    Expanded(
                      child: CustomButton(
                        text: 'Camera',
                        icon: Icons.camera_alt,
                        onPressed: controller.pickImageFromCamera,
                      ),
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: CustomButton(
                        text: 'Gallery',
                        icon: Icons.photo_library,
                        onPressed: controller.pickImageFromGallery,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 20),

                // Process button
                CustomButton(
                  text: 'Process Image',
                  icon: Icons.search,
                  onPressed:
                      controller.selectedImage.value != null
                          ? controller.processImage
                          : null,
                  isDisabled: controller.selectedImage.value == null,
                ),
              ],
            ),
          ),
        );
      }),
    );
  }
}
