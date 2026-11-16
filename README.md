# Aerial Trajectory Forecasting

İHA (drone) görüntülerinde yaban hayatı ve kırsal trafik izleme için tespit, takip ve kısa vadeli yörünge tahmini yapan bir bilgisayarlı görü hattı.

Sistem her karede nesneleri bulur, kameranın kendi hareketini çıkarır, her nesneye kalıcı bir kimlik verir ve nesnenin 0,5 / 1,0 / 1,5 saniye sonra (30 FPS'te t+15, t+30, t+45 kare) nerede olacağını belirsizlik elipsleriyle birlikte tahmin eder.

## Nasıl çalışır

| Aşama | Sınıf | Görev |
|---|---|---|
| 1. Tespit | `Detector` | YOLOv8 `model.predict`; en-boy oranı korunur |
| 2. Kamera telafisi | `EgoMotionCompensator` | Maskelenmiş Shi-Tomasi köşeleri, Lucas-Kanade akışı, RANSAC homografi |
| 3. Takip | `EntityTracker` | IoU + merkez uzaklığı eşleştirme, Kalman filtresi; hiçbir tespit bastırılmaz |
| 4. Tahmin | `TrajectoryForecaster` | `TrajectoryGRU` (2 katman, 64 gizli birim); genç izler için sabit hız yedeği |
| 5. Çizim | `Visualizer` | Kutular, kimlikler, zemine kilitli izler, tahmin eğrisi, 2σ elipsler |

Tahmin ağı ölçekten bağımsız öznitelikler kullanır: `[dx/w, dy/h, dw/w, dh/h]`. Böylece irtifa ve kamera eğimi değişse de girdiler benzer kalır. Ağ, sabit hızlı uzatmanın üzerine bir düzeltme öğrenir; eğitilmemiş hali doğrusal tahmine eşittir.

## Depo yapısı

```
trajectory_common.py      Ortak çekirdek: kamera telafisi, öznitelikler, GRU, checkpoint
inference_pipeline.py     Video üzerinde tespit + takip + tahmin
train_trajectory_gru.py   VisDrone-VID / UAVDT üzerinde TrajectoryGRU eğitimi
tools/plot_training.py    Eğitim eğrileri
tools/evaluate_model.py   Doğrulama verisinde değerlendirme grafikleri
weights/                  Model ağırlıkları (git'e eklenmez)
docs/                     Grafikler ve ekran görüntüleri
```

## Kurulum

```bash
git clone <depo-adresi>
cd aerial-trajectory-forecasting
pip install -r requirements.txt
```

Python 3.10 veya üstü gerekir. GPU zorunlu değildir.

## Kullanım

Ağırlıkları `weights/` klasörüne koy:

- `weights/detector.pt`: YOLOv8 tespit ağırlığın
- `weights/trajectory_gru.pth`: eğitilmiş tahmin modeli (yoksa sabit hız tahmini kullanılır)

```bash
python inference_pipeline.py --source input.mp4 --output output.mp4
```

| Parametre | Varsayılan | Açıklama |
|---|---|---|
| `--weights` | `weights/detector.pt` | YOLOv8 ağırlığı |
| `--gru` | `weights/trajectory_gru.pth` | TrajectoryGRU checkpoint'i |
| `--source` | `input.mp4` | Girdi videosu |
| `--output` | `output.mp4` | Çıktı videosu |
| `--ego-max-dim` | `960` | Akış hesabı için uzun kenar (0 = yerel çözünürlük) |
| `--no-preview` | kapalı | Önizleme penceresini açmaz |

Çıktıda sarı eğriler GRU tahminini, gri eğriler sabit hız yedeğini gösterir. Sağ üstteki `EGO OK` kamera telafisinin çalıştığını, `EGO HOLD` o karede homografinin reddedildiğini belirtir.

## Eğitim

Veri setleri depoya dahil değildir; kendi kaynaklarından indirilmelidir.

```bash
python train_trajectory_gru.py \
  --visdrone-root <VisDrone2019-VID-train> <VisDrone2019-VID-val> \
  --uavdt-images <UAV-benchmark-M> \
  --uavdt-gt <UAV-benchmark-MOTD_v1.0/GT> \
  --output weights/trajectory_gru.pth
```

- Her VisDrone kökü `sequences/` ve `annotations/` klasörlerini içermelidir.
- UAVDT etiketleri `<dizi>_gt_whole.txt` biçiminde olmalıdır.
- Homografiler `homography_cache/` altına kaydedilir; ikinci çalıştırma bu adımı atlar.
- Eğitim/doğrulama ayrımı dizi bazında yapılır (%15 doğrulama).
- Epoch geçmişi `weights/trajectory_gru.history.json` dosyasına yazılır.

Grafikler:

```bash
python tools/plot_training.py weights/trajectory_gru.history.json --out docs
python tools/evaluate_model.py --checkpoint weights/trajectory_gru.pth \
  --visdrone-root <VisDrone2019-VID-train> <VisDrone2019-VID-val> \
  --uavdt-images <UAV-benchmark-M> --uavdt-gt <UAV-benchmark-MOTD_v1.0/GT> --out docs
```

`plot_training.py` ayrıca konsol çıktısının yapıştırıldığı bir `.txt` dosyasını da okur.

## Sonuçlar

VisDrone-VID (train + val) ve UAVDT üzerinde 97 eğitim, 17 doğrulama dizisi; 310.959 eğitim ve 54.174 doğrulama örneği. Hatalar nesnenin kendi kutu boyutu birimindedir; kaydedilen model 17. epoch'a aittir.

| Ufuk | TrajectoryGRU | Sabit hız | İyileşme |
|---|---|---|---|
| t+15 | 0.190 | 0.229 | %17 |
| t+30 | 0.396 | 0.502 | %21 |
| t+45 | 0.666 | 0.856 | %22 |

## Sınırlamalar

- Sonuçlar temiz etiket kutularıyla ölçülmüştür; canlı YOLO tespitlerinde hata daha yüksek olur.
- Eğitim verisi araç ve yaya ağırlıklıdır, hayvan içermez. Yaban hayatı görüntülerinde kendi verinizle ek eğitim gerekebilir.
- Tahminler piksel cinsindendir; metreye çevirmek için irtifa ve kamera bilgisi gerekir.
- Ufuklar kare sayısıyla tanımlıdır; 30 FPS dışındaki videolarda gerçek süre değişir.
- Tespit `max_det=25` ile sınırlıdır; yoğun sürü ve kalabalıklarda bu sınır aşılabilir.

## Veri setleri

- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset): yalnızca akademik kullanım için sunulmaktadır.
- [UAVDT](https://sites.google.com/site/daviddo0323/projects/uavdt): kullanım koşulları için veri setinin kendi sayfasına bakın.

Bu depo veri setlerini ve onlardan türetilmiş görüntüleri içermez.
