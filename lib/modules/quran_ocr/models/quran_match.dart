class QuranMatch {
  final String extractedText;
  final String verifiedText;
  final int similarity;
  final String source;
  final int sourceCount;
  final List<String> sources;
  final List<String> allExtractedTexts;
  final String surahName;
  final String surahNameAr;
  final String surahNum;
  final String verseNum;
  final String verseTranslation;
  final String globalVerseNum;

  QuranMatch({
    required this.extractedText,
    required this.verifiedText,
    required this.similarity,
    required this.source,
    required this.sourceCount,
    required this.sources,
    required this.allExtractedTexts,
    required this.surahName,
    required this.surahNameAr,
    required this.surahNum,
    required this.verseNum,
    required this.verseTranslation,
    required this.globalVerseNum,
  });

  factory QuranMatch.fromJson(Map<String, dynamic> json) {
    return QuranMatch(
      extractedText: json['extracted_text'] ?? '',
      verifiedText: json['verified_text'] ?? '',
      similarity: json['similarity'] ?? 0,
      source: json['source'] ?? '',
      sourceCount: json['source_count'] ?? 0,
      sources: List<String>.from(json['sources'] ?? []),
      allExtractedTexts: List<String>.from(json['all_extracted_texts'] ?? []),
      surahName: json['surah_name'] ?? '',
      surahNameAr: json['surah_name_ar'] ?? '',
      surahNum: json['surah_num']?.toString() ?? '',
      verseNum: json['verse_num']?.toString() ?? '',
      verseTranslation: json['verse_translation'] ?? '',
      globalVerseNum: json['global_verse_num']?.toString() ?? '',
    );
  }
}
