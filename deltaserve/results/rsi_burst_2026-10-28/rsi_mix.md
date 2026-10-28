# Mixed RSI traffic (ACE formula + finer, TraceLab timing) + 4 FT rounds

| load | system | request p50 / p90 / p99 (s) | held | TTFT p50 / p99 (ms) | max-TBT p50 / p99 (ms) | agent task p50 / p90 (s) | round wait (s) per round | mean | all rounds done (s) | FT share | correct formula / finer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| real trace timeline, 24 h window (1.15 rps, 10 s CV 1.88) | seq | 3.577 / 5.948 / 29.281 | 73 | 75.0 / 754.0 | 84.0 / 282.0 | 6.553 / 30.621 | 41.0, 39.4, 42.1, 39.4 | 40.5 | 609.4 | 0.271 | 207/288 / 159/574 |
| real trace timeline, 24 h window (1.15 rps, 10 s CV 1.88) | coguard | 3.94 / 7.904 / 12.68 | 0 | 105.0 / 558.0 | 87.0 / 331.0 | 9.575 / 28.768 | 58.6, 58.6, 63.2, 39.8 | 55.1 | 609.8 | 0.271 | 205/288 / 163/574 |
| real trace timeline, 24 h window (1.15 rps, 10 s CV 1.88) | tt | 23.648 / 63.896 / 78.639 | 0 | 802.0 / 3784.0 | 1185.0 / 1962.0 | 36.184 / 82.068 | 85.1, 84.1, 146.5, 58.0 | 93.4 | 628.2 | 0.26 | 213/288 / 161/574 |
| real trace timeline, 24 h window (1.15 rps, 10 s CV 1.88) | (standalone FT, from seq) | 5820 tok/s |||||||||||
| real trace timeline, 12 h window (1.78 rps, 10 s CV 1.26) | seq | 3.585 / 38.789 / 49.397 | 418 | 70.0 / 2753.0 | 83.0 / 576.0 | 9.658 / 46.902 | 44.0, 39.6, 44.6, 42.3 | 42.6 | 612.3 | 0.255 | 310/446 / 255/890 |
| real trace timeline, 12 h window (1.78 rps, 10 s CV 1.26) | coguard | 3.906 / 11.82 / 18.302 | 0 | 70.0 / 662.0 | 85.0 / 506.0 | 7.217 / 27.93 | 59.0, 39.8, 69.9, 73.9 | 60.7 | 643.9 | 0.255 | 310/446 / 241/890 |
| real trace timeline, 12 h window (1.78 rps, 10 s CV 1.26) | tt | 17.533 / 99.042 / 130.642 | 0 | 480.0 / 4069.0 | 1120.0 / 1884.0 | 25.864 / 76.736 | 100.9, 43.2, 130.5, 144.7 | 104.8 | 714.7 | 0.255 | 314/446 / 248/890 |
| real trace timeline, 12 h window (1.78 rps, 10 s CV 1.26) | (standalone FT, from seq) | 5715 tok/s |||||||||||
