Kullanılan Kütüphaneler: json, os, nltk, zemberek.
indirme linkleri:
	-- zemberek = https://pypi.org/project/zemberek-python/
	-- nltk = https://pypi.org/project/nltk/
	

1- "collocation_by_freq.py" dosyasını çalıştırmak için dökümanlar "documents/docs/" yolunun altında olmalıdır.

2- Kodun çalışabilmesi için gerekli kütüphaneler kurulmalıdır.
	2.1- # you need to run this if this is first time trying import stopwords.
		#import nltk
		#nltk.download('stopwords')
		#nltk.download('punkt')

3- 27.851 dosya olmasından dolayı sonuçların çıkarılması uzun zaman alabiliyor.
	3.1- O yüzden "collocation_by_freq.py" dosyası içindeki get_text() methodu içinde yorum içine alınmış
		sonuç alınmasını hızlandırmak için dosya sayısının ayarlanabileceği kod satırları bulunmaktadır.
  