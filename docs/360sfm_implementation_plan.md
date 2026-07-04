# 360 SfM Research Implementation Plan

作成日: 2026-07-04

対象PDF: `docs/SphereSfM_360SfM_技術調査レポート_2026-07-04.pdf`

## 目的

長時間の360度動画から安定したカメラ姿勢を推定し、COLMAP sparse modelを経由して3D Gaussian Splattingへ接続する。既存のSphereSfMをbaselineとして残しつつ、COLMAP 4.1の`panorama_sfm` + `global_mapper`を第一候補として実装・比較する。

## PDFからの主要方針

優先して比較する方式は次の3つ。

1. `panorama_sfm` + Global Mapper
2. `panorama_sfm` + Incremental Mapper
3. 既存SphereSfM

第二段階の研究候補は、原因切り分け後に追加する。

- matchingが弱い場合: EDM
- Global SfM backendを比較したい場合: MGSfM + Virtual Rig
- 商用エンジン比較: Metashape Spherical
- ERP直接処理の軽量比較: COLMAP native `EQUIRECTANGULAR`

## 現在の実装との対応

既存パイプラインは `Stage` を差し替える構成になっており、SfM方式は `registry.py` の `SFM` に追加すれば `--sfm` で選択できる。

既存の重要なファイル:

- `main.py`: CLI、stage順序、`--sfm` 選択
- `registry.py`: `SFM` の登録場所
- `pipeline/context.py`: `frames_dir`, `cubemap_dir`, `masked_dir`, `sparse_dir`
- `pipeline/stages/sfm/colmap.py`: 標準COLMAPの既存Stage
- `pipeline/stages/sfm/spheresfm.py`: SphereSfM baseline
- `pipeline/stages/preprocessing/cubemap_splitter.py`: 既存のERP to perspective crop

重要な設計判断:

`panorama_sfm` 系は、既存の `cubemap/` をそのまま入力にするのではなく、原則として `frames/` のERP画像を入力にする別Stageとして実装する。理由は、PDFで重要視されている「同一360度フレームから作られたPerspective ViewをRigとして扱う」関係を、単純な分割画像投入では落とす可能性があるため。

## Phase 0: 実行環境とCOLMAP 4.1確認

目的: 実装前にローカルCOLMAPの実際のCLIを固定する。

実装タスク:

- `colmap version` を実行して4.1.0以上を要求するチェックを追加する。
- `colmap help`, `colmap global_mapper -h`, `colmap view_graph_calibrator -h` をログに残す調査スクリプトを作る。
- `panorama_sfm` がCOLMAP本体コマンドなのか、`scripts/python` のexampleなのか、ローカル配布物で確認する。
- Windowsで `pycolmap` が必要な場合の導入手順を `docs/` に記録する。

成果物:

- `docs/colmap_41_environment.md`
- ローカルCOLMAP 4.1のコマンド可用性チェック

## Phase 1: 公平比較の土台を作る

目的: 方式差だけを測れるように、入力フレーム、マスク、評価指標を固定する。

実装タスク:

- 実験単位を表す `ExperimentManifest` を追加する。
- 入力動画、抽出fpsまたは間引き間隔、frame filter条件、mask条件、SfM方式、COLMAP versionをJSONに保存する。
- `output/<experiment>/metrics.json` を出力する評価Stageまたは評価コマンドを追加する。
- `colmap model_analyzer` の出力を保存し、登録画像数、点数、観測数、再投影誤差を集計する。
- 軌道確認用に `images.txt` / `images.bin` からカメラ中心をCSVに出す。

成果物:

- `pipeline/stages/evaluation/colmap_metrics.py`
- `utils/colmap_model.py`
- `output/<run>/experiment_manifest.json`
- `output/<run>/metrics.json`
- `output/<run>/camera_trajectory.csv`

最初に測る条件:

- `fps=0.5`, `fps=1.0`, `frame_extraction_mode=frames` で数条件
- maskingなし
- `person.sky` maskあり
- 既存SphereSfM
- 標準COLMAP incremental

## Phase 2: `panorama_sfm` Stageを追加

目的: PDFの第一候補である `panorama_sfm` + Global / Incremental をCLIから実行できるようにする。

実装タスク:

- `config.py` に `PanoramaSFMSettings` を追加する。
- `config/default.ini` に `[PanoramaSFMSettings]` を追加する。
- `pipeline/stages/sfm/colmap_panorama.py` を新規作成する。
- `registry.py` に以下を追加する。
  - `panorama_global`
  - `panorama_incremental`
- `main.py` の `--set` override対象に `PanoramaSFMSettings` を追加する。
- `PanoramaSFMStage` は `ctx.frames_dir` を優先入力にする。
- 出力は必ず `ctx.work_dir / "sparse"` にCOLMAP sparse形式で揃える。

想定設定:

```ini
[PanoramaSFMSettings]
mapper = global
run_view_graph_calibrator = True
camera_model = OPENCV
num_virtual_views = 8
virtual_view_fov = 90
use_gpu = True
keep_intermediate = True
```

実装上の分岐:

- ローカルCOLMAPに `panorama_sfm` コマンドがある場合: 直接サブコマンドを呼ぶ。
- example scriptとして提供される場合: script pathを設定で受け取り、Python/pycolmap経由で呼ぶ。
- どちらもない場合: Phase 2では失敗を明示し、Phase 3で独自Virtual Rig生成へ進む。

検証:

- fake `run()` を使って、Global時に `view_graph_calibrator` と `global_mapper` が呼ばれることをユニットテストする。
- Incremental時に `mapper` が呼ばれることをユニットテストする。
- `ctx.sparse_dir` が設定されることをテストする。

## Phase 3: Virtual Rigを内製できるようにする

目的: `panorama_sfm` exampleに依存しすぎず、MGSfMや他backendにも渡せる中間形式を作る。

実装タスク:

- `CubemapSplitter` が作る `.cubemap-manifest.json` を拡張し、各cropの元ERPフレーム、yaw、pitch、FOV、仮想カメラIDを保存する。
- `pipeline/stages/preprocessing/virtual_rig_splitter.py` を追加する。
- COLMAP databaseへrig/frame/sensor関係を書き込む補助を調査・実装する。
- 最小実装では、公式 `rig_configurator` または `pycolmap` を優先する。

成果物:

- `cubemap/.rig-manifest.json`
- `--sfm virtual_rig_global`
- `--sfm virtual_rig_incremental`

注意:

このPhaseは難所。まずPhase 2で公式 `panorama_sfm` 経路を動かしてから着手する。

## Phase 4: COLMAP native ERPを軽量比較として追加

目的: 公式のEQUIRECTANGULAR camera modelを、速いが精度が落ちる可能性のある比較対象として測る。

実装タスク:

- `pipeline/stages/sfm/colmap_equirect.py` を追加する。
- `ImageReader.camera_model = EQUIRECTANGULAR` を設定する。
- 入力は `ctx.frames_dir` を使う。
- mapperは `global` / `incremental` を設定で切替可能にする。

登録名:

- `colmap_equirect_global`
- `colmap_equirect_incremental`

比較価値:

- `panorama_sfm` より速いか
- 長時間周回で破綻しやすいか
- 3DGS下流品質に差が出るか

## Phase 5: EDMをmatching強化として追加

目的: SfMの破綻原因が特徴対応にある場合に、ERP向けdense matchingを試す。

実装タスク:

- EDMの公式実装・重み・ライセンスを確認する。
- `pipeline/stages/matching/edm.py` を追加するか、SfM Stage内の事前matching stepとして追加する。
- 出力matchesをCOLMAP databaseへimportする経路を作る。
- `matches_importer` / database直接書き込み / pycolmap のどれが安全か検証する。

登録案:

- `--sfm edm_panorama_global`
- `--sfm edm_equirect_global`

着手条件:

- Phase 2/4の結果で、登録失敗区間がmatching不足と判断できること。

## Phase 6: MGSfM + Virtual Rigを追加

目的: Global SfM backendの違いを評価する。

実装タスク:

- MGSfMの入力がCOLMAP databaseであることを確認する。
- Phase 3のVirtual Rig databaseを流用する。
- `pipeline/stages/sfm/mgsfm.py` を追加する。
- MGSfM出力をCOLMAP sparseとして検証する。

着手条件:

- `panorama_sfm + global_mapper` で軌道が折れる、または周回の閉じ方が不安定であること。

## Phase 7: Metashape比較

目的: 商用アライメントエンジンとの外部比較を行う。

実装タスク:

- 自動化可能範囲を確認する。
- `pipeline/stages/sfm/metashape.py` は最初はstubでもよい。
- GUI手順またはPython API手順を `docs/metashape_workflow.md` に記録する。
- COLMAP export後、同じ評価Stageへ流す。

着手条件:

- 研究比較として外部基準が必要になった時点。

## 評価プロトコル

同じ入力セットで最低限以下を比較する。

| 実験ID | SfM方式 | 入力 | mask | mapper | 目的 |
|---|---|---|---|---|---|
| A1 | SphereSfM | current cubemap | none | incremental | 既存baseline |
| A2 | SphereSfM | current cubemap | person.sky | incremental | mask効果 |
| B1 | panorama_sfm | ERP frames | none | global | 第一候補 |
| B2 | panorama_sfm | ERP frames | none | incremental | mapper差分 |
| B3 | panorama_sfm | ERP frames | person.sky | global | mask効果 |
| C1 | native ERP | ERP frames | none | global | 軽量比較 |
| D1 | MGSfM | virtual rig | best mask | global | 第二段階 |
| E1 | EDM + SfM | ERP frames | best mask | global | matching強化 |

評価指標:

- 登録画像数 / 入力画像数
- sparse点数
- mean / median reprojection error
- registrationが途切れる時刻
- 周回軌道の閉じ方
- 点群の二重化、傾き、崩壊
- 3DGS学習後のPSNR/SSIM/LPIPS
- 目視レンダリング品質
- 処理時間、RAM/VRAM、失敗率

## 実装順序

1. Phase 0: COLMAP 4.1環境確認
2. Phase 1: 評価とmanifest整備
3. Phase 2: `panorama_global` / `panorama_incremental`
4. 3方式比較: SphereSfM / panorama global / panorama incremental
5. 最良2方式を3DGSまで流す
6. 必要ならPhase 4のnative ERPを追加
7. 問題原因に応じてPhase 5 EDMまたはPhase 6 MGSfM
8. 論文・発表用にMetashape比較をPhase 7で追加

## 最初の実装PRの推奨範囲

最初のPRは小さく切る。

- `PanoramaSFMSettings`
- `PanoramaSFMStage`
- `registry.py` 登録
- `main.py` override対応
- fake command unit tests
- `docs/colmap_41_environment.md`

このPRではEDM/MGSfM/Metashapeには触らない。最初に公式COLMAP経路を通し、既存SphereSfMと公平に比較できる状態を作る。

## リスク

- COLMAP 4.1の `panorama_sfm` が環境によってサブコマンドではなくPython exampleとして提供される可能性がある。
- Global Mapperは良い焦点距離priorに依存するため、`view_graph_calibrator` または明示的なintrinsics設定が必要になる可能性が高い。
- 既存 `CubemapSplitter` の出力を使うだけではRig情報が欠落し、PDFの推奨構成を正しく再現できない可能性がある。
- 15分動画を一括投入すると計算量と誤対応が増えるため、最初は短区間・複数fpsで切り分ける。
- READMEや一部コメントが文字化けしているため、研究手順ドキュメントは新規MarkdownにUTF-8で残す。

## 直近の実行例

既存baseline:

```powershell
python main.py C:\data\room.mp4 output\spheresfm_baseline --masker none --sfm spheresfm
```

実装後の第一候補:

```powershell
python main.py C:\data\room.mp4 output\panorama_global --masker none --sfm panorama_global
```

実装後の比較対象:

```powershell
python main.py C:\data\room.mp4 output\panorama_incremental --masker none --sfm panorama_incremental
```

