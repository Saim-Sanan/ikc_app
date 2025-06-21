class TopCandidate {
  final String source;
  final String text;
  final int arabicChars;
  final String? surahNum;
  final String? verseNum;
  final String? surahName;
  final String? surahNameAr;
  final String? verseTranslation;
  final String? globalVerseNum;
  final int? similarity;
  final String? verifiedText;

  TopCandidate({
    required this.source,
    required this.text,
    required this.arabicChars,
    this.surahNum,
    this.verseNum,
    this.surahName,
    this.surahNameAr,
    this.verseTranslation,
    this.globalVerseNum,
    this.similarity,
    this.verifiedText,
  });

  factory TopCandidate.fromJson(Map<String, dynamic> json) {
    return TopCandidate(
      source: json['source'] ?? '',
      text: json['text'] ?? '',
      arabicChars: json['arabic_chars'] ?? 0,
      surahNum: json['surah_num']?.toString(),
      verseNum: json['verse_num']?.toString(),
      surahName: json['surah_name'],
      surahNameAr: json['surah_name_ar'],
      verseTranslation: json['verse_translation'],
      globalVerseNum: json['global_verse_num']?.toString(),
      similarity: json['similarity'],
      verifiedText: json['verified_text'],
    );
  }
}
