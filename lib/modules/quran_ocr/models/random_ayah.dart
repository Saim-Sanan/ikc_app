class RandomAyah {
  final int number;
  final String text;
  final String translation;
  final String surah;
  final int numberInSurah;

  RandomAyah({
    required this.number,
    required this.text,
    required this.translation,
    required this.surah,
    required this.numberInSurah,
  });

  factory RandomAyah.fromJson(Map<String, dynamic> json) {
    final data = json['data'];
    final surah = data['surah'];
    
    return RandomAyah(
      number: data['number'] ?? 0,
      text: data['text'] ?? '',
      translation: data['translation'] ?? '',
      surah: surah['name'] ?? '',
      numberInSurah: data['numberInSurah'] ?? 0,
    );
  }
}
