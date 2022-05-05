#!/usr/bin/env python
# coding: utf-8
import pandas as pd
df = pd.read_excel('general_hukums.xlsx')
vucut_dokunulmazligina_karsi = []
for i in df["Suç"]:
    
    if i == " yaralama" or i == " basit yaralama" or i == " neticesi sebebiyle ağırlaşmış yaralama" or i == " basit kasten yaralama" or i == " kasten yaralama" or i == " taksirle yaralama":
        vucut_dokunulmazligina_karsi.append("Vücut dokunulmazlığına karşı suçlar")
    
    elif i == " tehdit" or i == " silahla tehdit" or i == " kişiyi hürriyetinden yoksun kılma" or i == " konut dokunulmazlığının ihlali" or i == " konut dokunulmazlığını bozma" or i == " şantaj" or i == " konut dokunulmazlığının ihlal" or i == " kişilerin huzur ve sükununu bozma" or i == " iş yeri dokunulmazlığını bozma" or i == " iş ve çalışma hürriyetinin ihlali" or i == " iş yeri dokunulmazlığını ihlali" or i == " konut dokunulmazlığını ihlal etme" or i == " iş yeri dokunulmazlığını ihlal":
        vucut_dokunulmazligina_karsi.append("Hürriyete karşı suçlar")
    
    elif i == " hakaret":
        vucut_dokunulmazligina_karsi.append("Şerefe karşı suçlar")
        
    elif i == " hırsızlık" or i == " nitelikli hırsızlık" or i == " mala zarar verme" or i == " hizmet nedeniyle güveni kötüye kullanma" or i == " dolandırıcılık" or i == " nitelikli dolandırıcılık" or i == " nitelikli yağma" or i == " yağma" or i == " marka hakkına tecavüz" or i == " suç eşyasının satın alınması veya kabul edilmesi" or i == " karşılıksız yararlanma" or i == " hakkı olmayan yere tecavüz" or i == " güveni kötüye kullanma" or i == " kamu kurum ve kuruluşlarının zararına dolandırıcılık" or i == " nitelikli hırsızlığa teşebbüs":
        vucut_dokunulmazligina_karsi.append("Mal varlığına karşı suçlar")
    
    elif i == " uyuşturucu madde ticareti yapma" or i == " kullanmak için uyuşturucu madde bulundurma" or i == " 4733 sayılı kanuna muhalefet" or i == " kenevir ekme" or i == "kullanmak için uyuşturucu madde bulundurma" or i == " 4733 sayılı yasaya muhalefet" or i == " 4733 sayılı kanuna muhalefet" or i == " uyuşturucu madde ticareti yapma veya sağlama" or i == " 4733 sayılı yasaya aykırılık":
        vucut_dokunulmazligina_karsi.append("Kamunun sağlığına karşı suçlar")
    
    elif i == " 5607 sayılı kanuna muhalefet" or i == " resmi belgede sahtecilik" or i ==  " 5607 sayılı kanuna aykırılık" or i == " 5607 sayılı yasaya muhalefet" or i == " sahte fatura düzenleme" or i == " 5607 sayılı yasaya aykırılık" or i == " özel belgede sahtecilik" or i == " sahte fatura kullanma" or i == " defter ve belgeleri gizleme" or i == " mühür bozma" or i == " resmi belgenin düzenlenmesinde yalan beyan" or i == " sahte fatura düzenlemek" or i == " defter" or i == " 5846 sayılı kanuna aykırılık" or i == " defter ve belge gizleme":
        vucut_dokunulmazligina_karsi.append("Kamu güvenine karşı suçlar")
    
    elif i == " görevi yaptırmamak için direnme" or i == " görevi kötüye kullanma" or i == " kamu malına zarar verme" or i == " zimmet" or i == " görevi kötüye kullanmak":
        vucut_dokunulmazligina_karsi.append("Kamu idaresinin güvenilirliğine ve işleyişine karşı suçlar")
    
    elif i == " silahlı terör örgütüne üye olma" or i == " 6136 sayılı yasaya aykırılık" or i == " 6136 sayılı yasaya muhalefet" or i == " 6136 sayılı kanuna aykırılık" or i == " 6136 sayılı kanun'a muhalefet" or i == " 6136 sayılı kanuna muhalefet" or i == " 6136 sayılı kanun'a aykırılık" or i == " genel güvenliği tehlikeye sokacak şekilde kasten silahla ateş etme":
        vucut_dokunulmazligina_karsi.append("Anayasal düzene ve bu düzenin işleyişine karşı suçlar")
        
    elif i == " trafik güvenliğini tehlikeye sokma" or i == " genel güvenliğin kasten tehlikeye sokulması":    
        vucut_dokunulmazligina_karsi.append("Genel tehlike yaratan suçlar")
        
    elif i == " cinsel taciz" or i == " çocuğun nitelikli cinsel istismarı" or i == " çocuğun cinsel istismarı" or i == " cinsel saldırı" or i == " nitelikli cinsel saldırı" or i == " beden veya ruh sağlığını bozacak şekilde çocuğun nitelikli cinsel istismarı" or i == " beden veya ruh sağlığını bozacak şekilde çocuğun cinsel istismarı":
        vucut_dokunulmazligina_karsi.append("Cinsel dokunulmazlığa karşı suçlar")
    elif i == " kasten öldürme" or i == " kasten öldürmeye teşebbüs" or i == " taksirle öldürme":
        vucut_dokunulmazligina_karsi.append("Hayata karşı suçlar")
    elif i == " i̇ftira" or i == " başkasına ait kimlik veya kimlik bilgilerinin kullanılması" or i == " muhafaza görevini kötüye kullanma" or i == " hükümlü veya tutuklunun kaçması" or i == " başkasına ait kimlik veya kimlik bilgilerini kullanma" or i == " başkalarına ait kimlik veya kimlik bilgilerinin kullanılması" or i == " iftira" or i == " suç uydurma":
        vucut_dokunulmazligina_karsi.append("Adliyeye karşı suçlar")
    elif i == " i̇mar kirliliğine neden olma" or i == " 2863 sayılı kanuna aykırılık" or i == " 6831 sayılı kanuna aykırılık":
        vucut_dokunulmazligina_karsi.append("Çevreye karşı suçlar")
    elif i == " göçmen kaçakçılığı":
        vucut_dokunulmazligina_karsi.append("Göçmen kaçakçılığı ve insan ticareti")
    elif i == " çocuğun kaçırılması ve alıkonulması":
        vucut_dokunulmazligina_karsi.append("Aile düzenine karşı suçlar")
    elif i == " tefecilik" or i == " tefecilik yapmak":
        vucut_dokunulmazligina_karsi.append("Ekonomi, sanayi ve ticarete ilişkin suçlar")
    elif i == " fuhuş" or i == " müstehcenlik":
        vucut_dokunulmazligina_karsi.append("Genel ahlaka karşı suçlar")
    elif i == " banka veya kredi kartlarının kötüye kullanılması" or i == " bilişim sistemleri banka veya kredi kurumlarının araç olarak kullanılması suretiyle dolandırıcılık" or i == " 5809 sayılı kanuna aykırılık":
        vucut_dokunulmazligina_karsi.append("Bilişim alanında suçlar")
    else:
        vucut_dokunulmazligina_karsi.append("others")

df["YeniSuclar"] = vucut_dokunulmazligina_karsi
df.to_csv("latest_hukums_with_classes_csv_file1.csv")
