# picodaq — PicoScope 3416E 取得・QL・共通解析ライブラリ

TlBr単素子（ch A: anode, ch B: cathode）の波形を PicoScope 3416E MSO の rapid block で全数取得します。
取得しながらGUIでクイックルックし、HDF5に保存します。
波形整形などの処理部分は、QLと本解析で同じコードを使う構成にしています。

```
picodaq/
├── config_acquisition.yaml     設定（取得 + 波形処理 + QL表示）3416E 用
├── config_6402c.yaml           同じく 6402C 用（scope.driver: ps6000）
├── pyproject.toml
├── src/picodaq/
│   ├── config.py      設定読み込み（既定値とマージ）            ┐
│   ├── io.py          HDF5 書き出し / 読み込み (WaveformReader)  │ 本解析でも使う
│   ├── shaping.py     台形整形・波高・品質カット量・τ調整        │（GUI・スコープ不要）
│   ├── timing.py      タイムスタンプ・入力レート・ライブタイム    │
│   ├── spectrum.py    ヒストグラム・ピークフィット                │
│   ├── depth.py       CAR による深さ補正 (DepthCorrection)        │
│   ├── calibration.py 線源ライン表・エネルギー校正                 │
│   ├── events.py      イベントファイル (FITS, EventFile)          │
│   ├── carcal.py      オプションの2次元校正 (CarCalibration)      │
│   ├── plots.py       QA 図                                      ┘
│   ├── backends.py    PicoScope 3000E (pypicosdk) / シミュレータ / ドライバの選択
│   ├── ps6000.py      PicoScope 6402C など旧 6000 系 (ctypes で libps6000 を直接呼ぶ)
│   ├── acquisition.py 取得ループ（GUIの子プロセス / CLI）
│   ├── gui.py         QL GUI (PySide6 + pyqtgraph)
│   ├── hwcheck.py     実機テストラン
│   ├── info.py        HDF5ファイルの要約
│   └── analyze.py     本解析の CLI (picodaq-analyze)
├── analysis.yaml      本解析の設定例
├── notebooks/tlbr_analysis.ipynb   本解析ノートブック（ライブラリ利用版）
└── tests/             pytest（スコープ不要）
```

## 1. インストール

Python 3.10 以上が必要です。実機を使うときは、PicoSDK（ドライバ）が入っていることも前提です。

```bash
cd picodaq
python -m venv .venv && source .venv/bin/activate      # conda環境でも可
pip install -e ".[all]"       # numpy h5py pyyaml + PySide6 pyqtgraph pypicosdk scipy astropy matplotlib pytest
```

`-e` で入れておくと、ソースを直した内容がそのまま反映されます。
インストールすると次のコマンドが使えるようになります。

| コマンド | 内容 |
|---|---|
| `picodaq-gui` | QL GUI |
| `picodaq-acquire` | GUIなしの取得（旧 `run_acqusition.py` 相当） |
| `picodaq-check` | 実機テストラン |
| `picodaq-info` | HDF5ファイルの要約 |
| `picodaq-analyze` | 本解析（イベントファイル → 深さ補正 → エネルギー校正） |

## 2. テストラン（この順番で）

### 2-1. 単体テスト（スコープ不要、1分弱）
```bash
pytest
```
全部 passed になればOKです。テストの内容は次のとおりです。
- 台形整形がnotebookの実装と一致すること
- 波高の再現、τ調整、品質カット
- HDF5の往復（旧形式のファイルも読めること）
- タイムスタンプのラップアラウンドとリセット、ライブタイム
- 偽のpypicosdkを使った Stop / タイムアウト
- シミュレータを使った取得、Tune/Run のロック
- GUIのスモークテスト
- 本解析：シミュレーションした Ba/Eu/Cs で、イベントファイル → 深さ補正 → 校正が既知のライン位置を ±2 keV で再現すること、FITS の往復、CLI

### 2-2. シミュレータでGUIを触る
```bash
picodaq-gui --sim
```
Cs-137っぽいパルスが600 Hzで出ます。Tuneモードで閾値を変えて Apply すると、レートのグラフに縦線が入り、入力レートとライブ率が変わるのを確認してください。

### 2-3. 実機テストラン
パルサー（周波数が分かっているもの）を ch A に入れるのがおすすめです。線源でも動きます。

```bash
picodaq-check --pulser-hz 1000 --save --json check.json
```

| チェック | 見ること |
|---|---|
| first batch / data A,B | 波形が取れているか、電圧レンジが妥当か |
| trigger info | `get_trigger_info` が動くか（タイムスタンプとmissedTriggersが取れるか） |
| event spacing | イベント間隔の最小値。レコード長（250 µs）以上なら、プリトリガ分を溜め直すまで次のトリガを受け付けていない |
| timestamp across batches | RunBlock をまたいでもタイムスタンプが続いているか。**WARN（毎バッチでリセット）** の場合、レートとライブ率はバッチ内でしか出せない |
| missed during readout | バッチ間の読み出し中のトリガが missedTriggers に数えられているか。パルサーの場合は「周波数 × gap」程度になるはず |
| input rate vs pulser | 入力レートがパルサー周波数と ±2% で合うか。**ライブ率の計算が正しいかの検証になる** |
| threshold change | 閾値を変えたときにタイムスタンプがリセットされるか（記録しておくだけ） |
| batch timeout / stop | 1秒のタイムアウトと、取得途中のStopが効くか |
| simple loop | 旧来の `run_simple_rapid_block_capture` も動くか |

FAILが出たら、`check.json` と画面の出力を送ってください。
ポーリングのループ（`capture.loop: poll`）が動かない場合は、`loop: simple` にすれば元の挙動に戻ります（その場合Stopは効きにくくなります）。

### 2-4. 短い本番
```bash
picodaq-gui                      # Tuneで閾値を決める → Runに切り替えて数分
picodaq-info data/20261111/tlbr_00000.h5 --process 2000
```

### 2-5. 本解析（Ba-133 の既存データ）
```bash
picodaq-analyze tau ../data/20260909/run_Ba/data/tlbr_00000.h5
picodaq-analyze run analysis.yaml          # analysis/20260909/ に FITS と QA 図
```
`summary.txt` に、81 / 276 / 303 / 356 keV が残差 ±1 keV 程度で並べば OK です（6-6 参照）。

**データの置き場所**：`analysis.yaml` とノートブックは、リポジトリの隣にある `data/` を見ます（相対パス）。
```
TlBr/
├── picodaq/        ← このリポジトリ（git）
└── data/20260909/run_Ba/data/tlbr_00000.h5 …   ← 測定データ（git には入れない）
```
別の場所にあるときは、`analysis.yaml` の `data_root` を書き換えるか、環境変数で指定します：
`export PICODAQ_DATA=/path/to/data`（`analysis.yaml` とノートブックの両方に効きます）。
`analysis.yaml` の中の相対パスは、コマンドを実行した場所ではなく、`analysis.yaml` のあるフォルダが基準です。

## 3. GUI

- **Mode**
  - **Tune**：取得中に閾値・バッチ数・バッチタイムアウトを変え、**Apply** で次のバッチから反映します。保存は任意で、保存する場合は `tlbr_tune_NNNNN.h5` になります。
  - **Run**：保存は必須で、設定はロックされます（GUIで操作できず、取得プロセス側でも変更コマンドを拒否します）。Start時の閾値などは、そのままファイルの `Config` に記録されます。
- **Save to / Name**：保存先フォルダとファイル名の頭（`output.data_dir` と `output.base` の値が初期値）。止まっている間だけ変えられ、Start 時に反映されます。
  - ファイル名は Run なら `<Name>_NNNNN.h5`、Tune なら `<Name>_tune_NNNNN.h5`。NNNNN は既存のファイルを避けて自動で増えるので、上書きはしません。
  - Name に使えるのは英数字と `.` `_` `-` だけです（`/` や空白は不可）。フォルダには `%Y%m%d` などの日付の書式が使えます。
- **over range**（ステータス行の右端）：レンジを超えたイベントの数と割合をチャンネルごとに数えます。1 つでもあると、ステータス行が赤くなります。Tune のうちに、0 になるようにレンジを決めてください。振り切れた波形も保存はされ、`Trigger/Overflow` に印が付きます。
  - その下の太字の行に、止まっている間は **次に作るファイル**、取得中は **書き込み中のファイル**（フルパスと波形数）が出ます。1 GB でファイルが切り替わると表示も変わります。
- **Batch**：rapid blockの1回あたりのイベント数です。
- **Batch timeout**：これを過ぎたら、その時点までに取れた分を読み出します。低レートのときに、QLの更新とStopの応答をよくするためのものです。
- 表示内容：
  - 直近20波形
  - anode / cathode の波高スペクトル（本解析と同じ台形整形・フラットトップ平均）
  - C/A vs anode
  - 入力レート（captured + missed）と accepted レート
  - ライブ率（レートの下の行）
- QLのデータはGUIが追いつかなければ捨てますが、ディスクへの書き込みは止まりません。

### デッドタイムについて
rapid blockで1つのトリガを取った後、次のトリガを受け付けるまでに、以下の時間がかかります。
1. ポストトリガ分（200 µs）の記録
2. 次のセグメントへの切り替えとトリガの再設定（**再アーム**、短い）
3. プリトリガ分（50 µs）の溜め直し

つまり、1イベントあたりおおよそレコード長の分だけ受け付けません。これに加えて、バッチ間の読み出し（USB転送、HDF5書き込み、QL計算）も不感時間になります。
この間に来たトリガは `missedTriggers` に数えられるので、ライブ率は `accepted / (accepted + missed)` で求めています。

## 4. GUIなしの取得
```bash
picodaq-acquire 30                     # 30分（最短1分）、Run、保存
picodaq-acquire --mode tune --no-save  # 保存せずにレートだけ見る
picodaq-acquire --sim                  # シミュレータ
```

## 5. HDF5フォーマット
| パス | 内容 |
|---|---|
| attrs | `PicoScopeModel`, `PicoScopeSerial`, `Config`(YAML), `RunMode`, `Waveforms`, `ActualSampleRateHz`, `TimestampUnitS` |
| `Time` | 時間軸 [s]（0 = トリガ位置） |
| `Waveforms/A`, `/B` | (n, 5000) float32 [V]、記録したままの極性 |
| `Trigger/Timestamp` | スコープのクロックのtick（× `TimestampUnitS` で秒）、56 bitでラップアラウンド |
| `Trigger/MissedTriggers` | 前のセグメントから取りこぼしたトリガの数 |
| `Trigger/Status` | `0x01000000` = タイムスタンプがリセットされた、`-1` = 情報なし |
| `Trigger/Overflow` | レンジを超えた（振り切れた）チャンネルのビット（A = 1、B = 2、C = 4、D = 8、0 = なし）。スコープが報告しない場合は attrs `OverflowAvailable = 0` |
| `BatchInfo` | (バッチ数, 4)：開始unix時刻、終了unix時刻、イベント数、閾値 [mV] |

旧 `run_acqusition.py` のファイル（`Time` と `Waveforms` だけ）も `WaveformReader` で読めます。

タイムスタンプはスコープ自身のクロックなので、加速器の時刻とは同期していません。
ビームパルスで弾く場合は、ビームのタイミング信号を同じレコードに記録するか、`BatchInfo` のunix時刻と組み合わせる必要があります。

## 6. 本解析

流れは **波形 (HDF5) → イベントファイル (FITS) → 深さ補正 → エネルギー校正 → ENERGY 列** です。
CLI、ノートブック、自分で書くスクリプトのどれも、同じクラスを使います。

### 6-1. CLI
```bash
picodaq-analyze tau    /path/run_Ba/data/tlbr_00000.h5        # τ と立ち上がり時間の確認
picodaq-analyze run    analysis.yaml                           # 以下を一括で
# 1段ずつ実行する場合
picodaq-analyze --analysis analysis.yaml events /path/run_Ba/data -o Ba133_events.fits
picodaq-analyze --analysis analysis.yaml depth  Ba133_events.fits -o depth.fits --plot depth.png
picodaq-analyze --analysis analysis.yaml calib  Ba133_events.fits:Ba133 Eu152_events.fits:Eu152 \
                                                --depth depth.fits -o calib.fits --plot calib.png
picodaq-analyze energy Ba133_events.fits Eu152_events.fits --depth depth.fits --calib calib.fits
```
`run` を実行すると `output_dir` に次のファイルができます。
- `<線源>_events.fits`（ENERGY 入り）
- `depth.fits`、`calib.fits`
- `depth.png`、`calib.png`、`spectra.png`
- `summary.txt`、`analysis_used.yaml`

イベントファイルが既にある場合は、整形をやり直さずに再利用します（`--force` で作り直し）。
設定項目は `analysis.yaml` のコメントを見てください。

### 6-2. ノートブック
`notebooks/tlbr_analysis.ipynb` は、元の notebook と同じ章立て（τ → 整形 → カット → スペクトル → CAR → 深さ補正 → 校正 → 保存）で、関数定義をライブラリの呼び出しに置き換えたものです。
最初のセルの `DATA` にデータの場所を書いてください。シミュレーションした Ba/Eu/Cs のデータで最後まで実行できることを確認済みです。

### 6-3. FITS の中身
| 拡張 | 中身 |
|---|---|
| `EVENTS` | `FILE_ID` `EVENT_ID` `TIME`[s] `EPOCH`、チャンネルごとの `AMP_X`[V] `BASE_X` `BLRMS_X` `BLSLOPE_X` `VMAX_X` `LATE_X`、`OVERFLOW`（上のビット）、`GOOD` `CAR`、補正後の `AMP_CORR`[V] `DEPTH_OK` `ENERGY`[keV]。header に整形とカットのパラメータ、HISTORY に processing の YAML |
| `FILES` | 元の HDF5 ごとの名前、イベント数、実時間、ライブタイム、入力レート、振り切れたイベント数 `N_OVERFLOW`（記録なしは −1） |

`GOOD` は、ベースライン RMS・ドリフト・遅れパルスのカットに加えて、anode か cathode が振り切れたイベントを除きます。振り切れも含めたいときは `cuts: {reject_overflow: false}`（`analysis.yaml`）か `ef.recompute_good(reject_overflow=False)` です。11/18 より前に取ったファイルには記録がないので、この除外は効きません。
| `DEPTH` | CAR スライスごとのピーク位置と、header に多項式係数・CAR 範囲・基準 |
| `CALIB` | 線源・ライン・波高・幅・残差・FWHM と、header に `MODEL` `C0` `C1` `C2` |

`depth.fits` と `calib.fits` は、それぞれ `DEPTH` / `CALIB` 拡張だけを持つ同じ形式です。
astropy なら `Table.read("x.fits", hdu="EVENTS")` で、そのまま表として読めます。

### 6-4. Python から
```python
from picodaq import EventFile, DepthCorrection, calibrate

ba = EventFile.from_waveforms("/path/run_Ba/data", processing=None)     # None: ファイルの設定
dc = DepthCorrection.fit(ba.amp("anode"), ba.column("CAR"), amp_window=(0.06, 0.14))
ba.apply_depth(dc)
cal = calibrate({"Ba133": ba.selected("AMP_CORR")}, model="linear")
ba.apply_calibration(cal)
ba.write("Ba133_events.fits"); cal.write("calib.fits")
print(cal.summary())
```

整形だけを使う場合：
```python
from picodaq import WaveformReader, load_config, processors_from_config
with WaveformReader("tlbr_00000.h5") as r:
    proc = processors_from_config(load_config(r.config)["processing"], r.channels, r.dt, r.trigger_index)
    feats = {ch: proc[ch].process_reader(r, ch) for ch in r.channels}  # amp, baseline, bl_rms, ...
    print(r.rate_summary())                                          # 実時間・入力レート・ライブ率
```

### 6-5. 元の notebook との対応
| notebook | picodaq |
|---|---|
| `load_file` / `read_waveforms` | `WaveformReader` / `ChannelProcessor.correct` |
| `trapezoid`, `flattop_window`, `extract_amplitude` | `TrapezoidShaper`（`.flattop`, `.amplitude`） |
| `tune_tau`, `fit_tail_tau`, `rise_time` | 同名（`tune_tau` は評価窓を変更） |
| `process_channel`, `quality_mask` | `ChannelProcessor.process`, `quality_mask`, `EventFile.from_waveforms` |
| `spectrum`, `gauss_lin`, `fit_peak` | `spectrum.histogram`, `gauss_lin`, `fit_peak`（`PeakFit` を返す） |
| `photopeak_vs_car`, `depth_correct` | `DepthCorrection.fit`, `.apply` |
| `gain = E / mu`（1点） | `calibrate` / `EnergyCalibration`（多点・3モデル）、オプションで `CarCalibration` |
| `np.savez_compressed(...amp.npz)` | `EventFile.write`（FITS） |

### 6-6. Ba-133 のデータ（2026-09-09、18,240 イベント）での確認
- 深さ補正後のピークは 81 / 276 / 303 / 356 keV で、直線との残差は ±1 keV 以内でした。ゲインはおよそ 3.0 keV/mV、オフセットは約 +3.5 keV です。Eu と Cs も加えて 11 本で校正すると、E = 5.1 + 3028 keV/V × a で、ほとんどのラインの残差は 1 keV 以内でした（964 keV だけ +3.6 keV、6-7 参照）。
- Ba の 356 keV と Cs の 662 keV で深さ補正の曲線（規格化したもの）を別々に作ると、1% 以内で一致しました。旧 `tlbr_00000.amp.npz` にある `gain_keV_per_V = 3487` は 662 keV（Cs）を前提にした値で、このファイルには合いません。
- 356 keV の FWHM は、旧 notebook の設定（flat 12 µs、15–85% を平均）だと約 5% でした。flat 16–20 µs にしてフラットトップの後半（0.6–0.9）だけを平均すると、約 2.4–2.9% に改善しました。τ を変えたときの差（A 45–100 µs、B 17–25 µs）はこれより小さいものでした。
- 実際のパルスは単一指数ではありません（例えばテールのフィットでは A 99 µs ですが、フラットトップが平らになる τ は 40–60 µs です）。そのため τ は droop ではなく分解能で選ぶのが確実で、ノートブックの §5 がそのためのスキャンです。
- CAR が 0.5 より小さい領域は、補正曲線が急で 3 次多項式では追い切れず、自動的に範囲外になりました（このファイルでは全体の約 4 割が範囲内）。

### 6-7. オプション：2次元校正（CAR × エネルギー）
`analysis.yaml` で `calibration.method: car2d` にすると、2段方式（深さ補正 → 校正）に加えて、次のモデルを全線源・全ライン・全 CAR スライスのピーク位置に同時にフィットします。

```
a = off(CAR) + E × gain(CAR)        a: anode の波高 [V], E: エネルギー [keV]
```

off と gain は CAR の多項式（既定は1次と3次）です。各イベントのエネルギーは、測った a と CAR からこの式を逆に解いて出します。2段方式の結果は、ピークを探す初期値としてだけ使います。

出力は次のとおりです。
- `carcal.fits`：`CARCAL` 拡張
- `carcal.png`：gain(CAR)、off(CAR)、ライン × CAR の残差
- `spectra_car2d.png`
- `summary.txt`：2つの方式での各ラインの重心と FWHM の比較表

イベントファイルの `ENERGY` は 2次元校正の値になります。Python からは `CarCalibration.fit(...)` と `EventFile.apply_carcal(cc)` で使えます。

2026-09-09 の Ba / Eu / Cs（合わせて約 11.5 万イベント、flat 16 µs、0.6–0.9 の窓）での比較：

| ライン [keV] | 2段方式 重心 / FWHM | 2次元 重心 / FWHM |
|---|---|---|
| Ba 81.0 | 82.7 / 8.2 | 84.7 / 10.9 |
| Eu 121.8 | 122.1 / 7.5 | 122.3 / 7.5 |
| Eu 244.7 | 244.6 / 8.9 | 245.2 / 8.4 |
| Ba 356.0 | 356.1 / 8.5 | 356.2 / 8.7 |
| Cs 661.7 | 661.6 / 14.1 | 661.5 / 13.4 |
| Eu 778.9 | 779.7 / 19.1 | 778.5 / 15.4 |
| Eu 964.1 | 967.8 / 17.6 | 964.2 / 11.6 |

- 600 keV より上では、2次元校正の方が重心も FWHM も良くなりました。356 keV 以下はほぼ同じです。
- 81 keV は、フィットに使えた一番低いライン（122 keV）より下なので、外挿になり悪化しています。
- gain は CAR 0.5 → 1.3 で 0.28 → 0.345 V/MeV、オフセットは −0.4 → −3 mV でした。
- CAR の有効範囲は、2次元校正の方が少し狭くなります（0.53–1.31）。
- E² の項（`quadratic: true`）は効きませんでした。

既定は今のところ 2段方式です。高エネルギー（UVSOR の約 3 MeV など）を扱うときは、2次元校正を試す価値があります。

## 7. PicoScope 6402C で使う（Mac）

6402C は旧 6000 系の `ps6000` ドライバで動きます。pypicosdk はこのドライバに対応していないので、
`ps6000.py` が `libps6000` を ctypes で直接呼びます。GUI・CLI・解析はそのまま使えます。

### 7-1. 準備
1. Pico の Downloads ページから **PicoSDK for macOS** をインストールします（Apple Silicon なら ARM-64 版。PicoScope 7 アプリだけでは入りません）。
2. ドライバが入ったか確認します。
   ```bash
   ls /Library/Frameworks/PicoSDK.framework/Libraries/
   # libps6000/ があれば OK（中に libps6000.dylib）
   ```
   別の場所にある場合は、`config_6402c.yaml` の `scope.sdk_path` にそのフォルダを書きます。
3. 6402C は USB 3.0 ポートに付けてください。USB 2.0 でも動きますが（警告が出ます）、読み出しが遅くなり、その分デッドタイムが増えます。外部電源も必要です。
4. PicoScope 7 アプリを閉じておきます（同時には開けません）。

### 7-2. テストラン
```bash
pytest -q tests/test_ps6000.py                         # 偽ドライバと、C でビルドした模擬ライブラリでの確認
picodaq-check --config config_6402c.yaml --pulser-hz 1000 --save --json check6402.json
picodaq-gui   --config config_6402c.yaml
```
ライブラリが見つからないときは、探したパスの一覧つきのエラーになります。

### 7-3. 3416E との違い
| | 3416E (psospa) | 6402C (ps6000) |
|---|---|---|
| ADC | 10 bit（設定による） | 8 bit |
| サンプリング | 20 MS/s | 20 MS/s を指定すると 19.53 MS/s（51.2 ns）。実際の値がファイルに入ります |
| イベントごとのタイムスタンプ | あり | **あり**（`ps6000GetTriggerInfoBulk`） |
| missedTriggers | あり | **なし** → 入力レートとライブ率は出せません。GUI では `n/a`、受理レートは出ます |
| 電圧レンジ | `V2` など | `mV10`…`V20`。8 bit なので **信号に合わせて狭いレンジ（例 `mV500`）** を推奨 |
| オフセット・帯域 | なし | `analog_offset_v`、`bandwidth: 20MHz`、`coupling: DC_1M / AC / DC_50R` |

`picodaq-check` では、6402C でできない項目（missedTriggers、missed during readout、simple loop）は
「対象外」の INFO になります。FAIL にはなりません。高い閾値でのタイムアウト確認には、
トリガチャンネルのレンジの 95%（`mV500` なら 475 mV）が自動で使われます。

確認してほしいのは次の 2 点です（ドキュメントからは決められなかったところ）。
- **timestamp across batches**：RunBlock をまたいでタイムスタンプが続くか。続かない場合は WARN になり、時刻はバッチ内の相対時刻になります。
- パルサーを入れた場合の **accepted rate vs pulser**：パルサーの周期がレコード長（250 µs）より長ければ、受理レートがパルサー周波数と合うはずです。

ファイルには `PicoScopeModel = 6402C` と `MissedTriggersAvailable = 0` が入ります。
`MissedTriggers` の列は 0 で埋まっていますが、読み込み側（`WaveformReader.trigger_info()`）は `None` を返します。

## 8. 設定 (`config_acquisition.yaml`)
省略したキーは `src/picodaq/config.py` の既定値が使われます。主な項目：

- `capture.loop`：`poll`（既定）または `simple`
- `capture.batch_timeout_s`
- `output.data_dir`：`data/%Y%m%d` のように strftime の書式が使えます
- `processing`：極性、τ、整形時間、`flattop_window`、波高の取り方。QL と解析で共通で、各ファイルにも保存されます。
- `ql`：ヒストグラムのビンなど、表示用の設定
- `simulate`, `sim_rate_hz`, `sim_source`（`Ba133` `Eu152` `Cs137`）, `sim_gain_kev_per_v`
- `scope.driver`：`psospa`（既定、3000E）または `ps6000`（6402C など）。`ps6000` 用のキーは 7 章

## 9. 既知の制限・今後
- ポーリングのループは pypicosdk の内部関数（`_call_attr_function`, `_adc_to_`）を使っています。確認したのは pypicosdk 1.7.5 です。
- バッチ間の処理（HDF5 書き込みでおよそ 11 ms/32 イベント、QL でおよそ 5 ms）がデッドタイムになります。次のバッチを先にアームしてから書き込む方式にすると減らせます（未実装）。
- `ps6000.py` は実機ではまだ確認していません。関数の型と構造体の並びは Pico 公式の picosdk-python-wrappers に合わせ、C で書いた模擬ライブラリでテストしています。
- missedTriggers と、バッチをまたいだタイムスタンプの連続性の挙動は、Pico のガイド（6000A 系、同じ API ファミリ）に基づいて実装しています。3000E での挙動は `picodaq-check` で確認してください。
- `tune_tau` は、フラットトップのうち電荷収集が終わった後ろ側だけで droop を評価します（旧 notebook は先頭 +10% から）。理想的な CSA 波形では正しい τ を返しますが、実データは単一指数ではないので参考値です。
- 深さ補正は 3 次多項式です。CAR の小さい側まで使うには、モデルを変える必要があります（区分的な補間など）。
- 時間変化（分極）：新しいファイルには `TIME` と `EPOCH` があるので、ピーク位置を時間のビンごとに追えます。まだ関数にはしていません。
