# 360Gaussian Pipeline

360°動画 / 画像から、3D Gaussian Splatting 用の **COLMAP sparse 再構成**（カメラ位置・姿勢＋疎な点群）を生成するパイプライン。

> 3DGSトレーニング自体（Lichtfeld 等）は範囲外。このパイプラインの出力（COLMAP sparse）を各トレーナーに渡す。

---

## 1. これは何をするのか

```
360°動画 / 画像フォルダ
   │
   ├─[1] Frame Extraction   動画→フレーム抽出 (FFmpeg)
   ├─[2] Frame Filter       ブレ画像を除去 (任意)
   ├─[3] Cubemap Split      正距円筒→透視投影画像に分割
   ├─[4] Masking            人物・空などを除外 (任意)
   └─[5] SfM                カメラ位置推定 → COLMAP sparse 出力
            ↓
        output/sparse/0/  (cameras.bin, images.bin, points3D.bin)
            ↓
        3DGSトレーナーへ (手動)
```

各ステップ（=Stage）は差し替え可能。詳しくは「6. 研究で手法を入れ替える」を参照。

---

## 2. 必要なもの

| 種類 | 内容 | 入手 |
|---|---|---|
| **Python** | 3.10 以上（3.12 で動作確認済み） | Python公式サイト等からインストール |
| **Pythonライブラリ** | opencv-python, numpy, Pillow, tqdm | `pip install`（下記） |
| **外部ツール** | 使うStageに応じて（COLMAP / AutoMasker / RealityScan / FFmpeg / colmap_sphere） | 各自で用意。`config/local.ini` にパス記入 |

外部ツールは Python 環境とは**別物**。pip では入らない。使う手法のものだけ用意すればよい。

| Stage | 必要な外部ツール | パス設定元 |
|---|---|---|
| Frame Extraction | FFmpeg | `[ToolPaths] ffmpeg`（空欄ならPATH） |
| `--sfm spheresfm` | colmap_sphere.exe | `[ToolPaths] colmap_sphere`（空欄ならPATH） |
| `--sfm colmap` | COLMAP | `[COLMAPPaths] colmappath`（空欄ならPATH） |
| `--sfm realitycapture` | RealityScan | `[PostShotPaths] realitycapturepath` |
| `--masker automasker` | AutoMasker.exe | `[AutoMaskerPaths] automaskerpath` |

---

## 3. まっさらなWindows環境からのセットアップ

以下は、Pythonや外部ツールがまだ入っていないWindows環境を想定した手順。

### 3.1 PythonとGitをインストール

次のソフトウェアをインストールする。

- **Python 3.12（64-bit推奨）**
- **Git for Windows**（ZIPでプロジェクトを受け取る場合は省略可）

Pythonのインストーラーでは、可能なら **Add Python to PATH** を有効にする。
インストール後、新しいPowerShellを開いて確認する。

```powershell
python --version
git --version
```

`python` が見つからない場合は、Windowsの「アプリ実行エイリアス」を確認するか、
PythonをPATH付きで再インストールする。

### 3.2 プロジェクトを取得

Gitを使う場合:

```powershell
git clone <このリポジトリのURL>
cd 360toGaussianSplatting
```

ZIPで取得した場合は展開し、PowerShellでそのフォルダへ移動する。

```powershell
cd C:\path\to\360toGaussianSplatting
```

### 3.3 Python仮想環境を作成

```powershell
# 仮想環境を作成
python -m venv .venv

# 仮想環境を有効化
.\.venv\Scripts\Activate.ps1

# pipを更新して依存ライブラリをインストール
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

インストール確認:

```powershell
python -c "import cv2, numpy, PIL, tqdm; print('Python dependencies: OK')"
python -m unittest discover -v
```

テストがすべて `OK` になればPython側の準備は完了。

> **`Activate.ps1` が「スクリプトの実行が無効」で弾かれた場合**
> 一度だけ以下を実行して許可する:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```
> または有効化せず、フルパスで直接実行してもよい:
> ```powershell
> .\.venv\Scripts\python.exe main.py ...
> ```

### 3.4 外部ツールを用意

Pythonライブラリだけではパイプラインは完走しない。使用する処理に応じて
外部ツールを別途インストールする。

#### 最小動作確認（Cubemap生成まで）

必要なのは **FFmpeg** のみ。FFmpegをインストールし、次を確認する。

```powershell
ffmpeg -version
```

PATHへ追加しない場合は、後述の設定ファイルで `ffmpeg.exe` のフルパスを指定する。

#### COLMAP sparse生成まで行う推奨構成

最初は次の構成が比較的シンプル。

- FFmpeg
- 標準COLMAP
- マスキングなし（`--masker none`）

COLMAPをインストールし、PATHを設定した場合は次を確認する。

```powershell
colmap -h
```

PATHを使わない場合は、設定ファイルの
`[COLMAPPaths] colmappath` に実行ファイルのフルパスを記入する。

#### その他の構成

- `--sfm spheresfm` には別途 `colmap_sphere.exe` が必要。
- `--masker automasker` にはAutoMasker本体とモデルファイルが必要。
- AutoMaskerのGPU実行には、配布物が要求するNVIDIAドライバーやCUDA環境も必要。
- `--sfm realitycapture` はCOLMAP形式への変換が未実装のため、現在は完走しない。

つまり、外部ツールをまだ用意していない状態でデフォルトの
`python main.py ...` を実行すると、SphereSFMまたはAutoMaskerが見つからず停止する。

### 3.5 ローカル設定ファイルを作成

共有設定を直接書き換えず、ローカル用設定をコピーして使う。
`config/local.ini` はGit管理対象外。

```powershell
Copy-Item config\default.ini config\local.ini
notepad config\local.ini
```

標準COLMAPを使う最小構成では、少なくとも次を実環境に合わせる。
PATHから見つかるツールは空欄のままでよい。

```ini
[ToolPaths]
ffmpeg =

[COLMAPPaths]
colmappath =
```

PATHを使用しない例:

```ini
[ToolPaths]
ffmpeg = C:\Tools\ffmpeg\bin\ffmpeg.exe

[COLMAPPaths]
colmappath = C:\Tools\COLMAP\COLMAP.exe
```

### 3.6 段階的に動作確認

まずSfMやマスキングを動かさず、フレーム抽出とCubemap生成だけ確認する。

```powershell
python main.py C:\data\room.mp4 output\smoke `
  --config config\local.ini `
  --masker none `
  --stop-after cubemap
```

`output\smoke\frames` と `output\smoke\cubemap` に画像が生成されれば成功。

次に標準COLMAPで最後まで実行する。

```powershell
python main.py C:\data\room.mp4 output\room `
  --config config\local.ini `
  --masker none `
  --sfm colmap
```

最終的に次の3ファイルが生成されることを確認する。

```text
output\room\sparse\0\cameras.bin
output\room\sparse\0\images.bin
output\room\sparse\0\points3D.bin
```

### 仮想環境について（よくある疑問）

- `python main.py ...` は**仮想環境を自動で立ち上げない**。実行時にPATHが通っているPythonを使うだけ。
- だから毎回まず `.\.venv\Scripts\Activate.ps1` で有効化してから実行する（インストールは初回のみ）。
- 別のターミナルを開いたら、また有効化が必要。
- 終わるときは `deactivate`。

---

## 4. 設定ファイル

全パラメータは標準的なINI形式で管理する。`config/default.ini` はひな形として残し、
通常はコピーした `config/local.ini` を `--config` で指定する。

**最初にやること**: `config/local.ini` 内の外部ツールのパスを実環境に合わせる。

```ini
[COLMAPPaths]
colmappath = C:\path\to\colmap.exe        ; --sfm colmap を使うなら

[AutoMaskerPaths]
automaskerpath = C:\path\to\AutoMasker.exe ; --masker automasker を使うなら

[VideoSettings]
fps = 1                ; 1秒に1フレーム抽出
splits = 8             ; 1フレームを8方向の透視画像に分割
fovvalue = 90          ; 各透視画像の画角(度)
useframefilter = False ; ブレ除去のON/OFF
```

主要パラメータ早見表:

| セクション | キー | 意味 |
|---|---|---|
| `[VideoSettings]` | `fps` | 抽出フレームレート |
| | `splits` | 透視投影の分割数（多いほど画像が増える） |
| | `fovvalue` | 各透視画像の画角 |
| | `useframefilter` | ブレ画像フィルタON/OFF |
| | `framestokeep` | 残す割合 `50%` or 枚数 |
| `[AutoMaskerSettings]` | `keywords` | 除外対象 `person.sky.car` のようにドット区切り |
| `[SphereSFMSettings]` | `matcher_type` | `sequential`（動画向き）/`exhaustive` |
| | `max_num_features` | 特徴点の最大数 |

---

## 5. 実行方法

有効化済み（`(.venv)` 表示）の状態で:

```powershell
# 基本形（デフォルト: SphereSFM + AutoMasker）
python main.py 入力 出力フォルダ

# 例: 動画から、マスキングなし・COLMAPで
python main.py C:\data\room.mp4 output\room --masker none --sfm colmap
```

### よく使うオプション

| オプション | 説明 | 例 |
|---|---|---|
| `--sfm` | SfM手法 | `--sfm colmap` / `spheresfm` / `realitycapture` |
| `--masker` | マスキング手法 | `--masker none` / `automasker` / `pregenerated` |
| `--skip` | 特定ステップを飛ばす | `--skip extraction,filter` |
| `--stop-after` | 途中で止める | `--stop-after cubemap` |
| `--set` | 設定をその場で上書き | `--set VideoSettings.splits=12` |
| `--config` | 別の設定ファイル | `--config config/myexp.ini` |
| `-v` | 詳細ログ | |

ステップ名: `extraction` / `filter` / `cubemap` / `masking` / `sfm`

`--skip` で再開する場合、対応する出力フォルダ（例: `frames/`、
`cubemap/`）が既に存在する必要があります。見つからない場合は処理開始前に
エラーになります。

### 実践例

```powershell
# フレーム抽出と分割だけ試す（SfM前に画像を確認したい）
python main.py room.mp4 out\ --masker none --stop-after cubemap

# 抽出済みの out\ を使い、SfMだけやり直す
python main.py room.mp4 out\ --skip extraction,filter,cubemap

# 分割数とFOVを変えて比較実験
python main.py room.mp4 out_a\ --set VideoSettings.splits=8  --set VideoSettings.fovvalue=90
python main.py room.mp4 out_b\ --set VideoSettings.splits=12 --set VideoSettings.fovvalue=75
```

### 出力

```
出力フォルダ/
├── frames/    抽出された生フレーム
├── cubemap/   透視投影画像（SfMの入力）
├── masked/    マスク済み画像（マスキング使用時）
│   ├── images/
│   └── masks/
└── sparse/
    └── 0/     ← これを3DGSトレーナーに渡す
        ├── cameras.bin
        ├── images.bin
        └── points3D.bin
```

---

## 6. 研究で手法を入れ替える

このプロジェクトの主目的。**新しい手法 = 新しいStageクラス1つ**。

### 仕組み

- `pipeline/base.py` の `Stage` を継承し、`name` と `run(ctx)` を実装するだけ。
- `registry.py` の辞書に1行追加すると `--masker` / `--sfm` で選べるようになる。
- Stage間のデータ受け渡しは `PipelineContext`（`pipeline/context.py`）のパス（`frames_dir`, `cubemap_dir`, `masked_dir`, `sparse_dir`）経由。

### 新しいマスキング手法を追加する例

**手順1**: `pipeline/stages/masking/my_masker.py` を作る

```python
from config import Config
from pipeline.base import Stage
from pipeline.context import PipelineContext

class MyMasker(Stage):
    @property
    def name(self) -> str:
        return "My Masker"

    def run(self, ctx: PipelineContext) -> PipelineContext:
        src = ctx.cubemap_dir or ctx.frames_dir   # 入力画像
        out = ctx.stage_dir("masked") / "images"  # 出力先
        out.mkdir(parents=True, exist_ok=True)
        # ... ここで自分の手法を実装 ...
        ctx.masked_dir = out                      # 後続SfMが参照する
        return ctx
```

**手順2**: `registry.py` に1行追加

```python
from pipeline.stages.masking.my_masker import MyMasker

MASKING = {
    "automasker":   AutoMaskerStage,
    "none":         PassthroughMasker,
    "pregenerated": PregeneratedMasker,
    "my_masker":    MyMasker,          # ← 追加
}
```

**手順3**: 使う

```powershell
python main.py room.mp4 out\ --masker my_masker
```

SfM手法・前処理手法も同じ要領（`pipeline/stages/sfm/` `pipeline/stages/preprocessing/` に置き、`registry.py` の `SFM` / `EXTRACTOR` 等に登録）。

### 既存Stageの内部アルゴリズムだけ差し替えたい場合

クラスを継承してメソッドを上書きするのが楽:

| 差し替えたいもの | 上書きするメソッド | ファイル |
|---|---|---|
| ブレ判定の指標 | `FrameFilter.score_frame()` | `preprocessing/frame_filter.py` |
| 投影アルゴリズム | `CubemapSplitter.reproject()` | `preprocessing/cubemap_splitter.py` |

---

## 7. ディレクトリ構成

```
360gaussian-pipeline/
├── README.md                ← この資料
├── main.py                  ← 実行エントリポイント / CLI
├── registry.py              ← ★手法の登録場所（差し替えはここ）
├── config.py                ← 設定の型定義＋INI読み込み
├── requirements.txt
├── config/
│   └── default.ini          ← パラメータ・外部ツールパス
├── pipeline/
│   ├── base.py              ← Stage 抽象クラス
│   ├── context.py           ← Stage間のデータ受け渡し
│   ├── orchestrator.py      ← Stageを順に実行
│   └── stages/
│       ├── preprocessing/   ← 抽出・フィルタ・分割
│       ├── masking/         ← マスキング各手法
│       └── sfm/             ← SfM各手法
└── utils/
    └── process.py           ← 外部コマンド実行（ログ表示付き）
```

---

## 8. 注意・既知の制約

- **外部ツールのCLIは要確認**: `automasker.py` / `spheresfm.py` / `realitycapture.py` の引数は、各ツールの実際の `--help` と照合してから使うこと（コード内に該当の `NOTE:` コメントあり）。
- **RealityCapture→COLMAP変換は未実装**: `realitycapture.py` の `_convert_to_colmap()` はスタブ。RC出力形式を確認後に実装が必要。
- **マスキングはオプション**: 動的物体（人・車）が無いシーンなら `--masker none` でよい。
- **入力が画像フォルダの場合**: 元画像は出力先の `frames/` にコピーされてから処理される（元データは保護される）。
```
