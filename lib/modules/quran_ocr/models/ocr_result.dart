import 'package:ikc_app/modules/quran_ocr/models/quran_match.dart';
import 'package:ikc_app/modules/quran_ocr/models/top_candidate.dart';

class OcrResult {
  final bool success;
  final String processingTime;
  final int matchCount;
  final List<QuranMatch> matches;
  final List<TopCandidate> topCandidates;

  OcrResult({
    required this.success,
    required this.processingTime,
    required this.matchCount,
    required this.matches,
    required this.topCandidates,
  });

  factory OcrResult.fromJson(Map<String, dynamic> json) {
    return OcrResult(
      success: json['success'] ?? false,
      processingTime: json['processing_time'] ?? '',
      matchCount: json['match_count'] ?? 0,
      matches:
          (json['matches'] as List<dynamic>?)
              ?.map((match) => QuranMatch.fromJson(match))
              .toList() ??
          [],
      topCandidates:
          (json['top_candidates'] as List<dynamic>?)
              ?.map((candidate) => TopCandidate.fromJson(candidate))
              .toList() ??
          [],
    );
  }

  factory OcrResult.error(String errorMessage) {
    return OcrResult(
      success: false,
      processingTime: '',
      matchCount: 0,
      matches: [],
      topCandidates: [],
    );
  }
}
