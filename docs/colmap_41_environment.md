# COLMAP 4.1 環境確認（Phase 0）

作成日: 2026-07-05
対象: `docs/360sfm_implementation_plan.md` Phase 0 / Phase 2

## 調査方法

```powershell
python scripts/probe_colmap_env.py            # PATH上のcolmapを調査
python scripts/probe_colmap_env.py --colmap C:\path\to\colmap.exe
```

このスクリプトは `colmap help` の出力からバージョンとサブコマンド一覧を解析し、
`panorama_sfm` 経路に必要な機能の有無を報告する。COLMAPを入れ替えたら再実行し、
下の「現在のローカル環境」を更新すること。

## 現在のローカル環境（2026-07-05 実測）

```
NG  COLMAP version: 3.8 (required for panorama_sfm: >= 4.1)
NG  subcommand: panorama_sfm
NG  subcommand: view_graph_calibrator
NG  subcommand: global_mapper
OK  subcommand: mapper
NG  pycolmap: not installed
```

- PATH上のCOLMAP: **3.8** (Commit 0dfffe7, 2025-10-30, CUDA有効)
- `panorama_sfm` / `global_mapper` / `view_graph_calibrator` は**存在しない**
- `sphere_cubic_reprojecer` など通常の3.8系コマンドは利用可能
- `pycolmap` は未インストール

つまり現状のままでは `--sfm panorama_global` / `--sfm panorama_incremental` は
実行時に明示的なエラーで停止する（実装計画どおりの挙動）。

## panorama_sfm を有効化する2つの経路

### 経路A: COLMAP 4.1+ をインストール（推奨）

1. COLMAP 4.1以降（CUDA版）を入手してインストールする。
2. `config/default.ini` の `[COLMAPPaths] colmappath` にexeのフルパスを設定する
   （PATHに追加した場合は空欄のままでよい）。
3. `python scripts/probe_colmap_env.py` で `panorama_sfm` / `global_mapper` /
   `view_graph_calibrator` がOKになることを確認する。

この場合 `PanoramaSFMStage` はネイティブのサブコマンド経路を自動選択する。

### 経路B: pycolmap example スクリプト（fallback）

COLMAP本体に `panorama_sfm` サブコマンドが無い配布物では、同機能が
`colmap/python/examples/panorama_sfm.py` としてpycolmapベースのexampleで
提供される場合がある。

1. `pip install pycolmap`（Windows: wheelが提供されているPythonバージョンを使う。
   合わない場合はconda `conda install -c conda-forge pycolmap` が確実）
2. COLMAPソースの `python/examples/panorama_sfm.py` を取得する。
3. `config/default.ini` に設定する:

```ini
[PanoramaSFMSettings]
panorama_script = C:\path\to\colmap\python\examples\panorama_sfm.py
python_exe =                  ; 空欄=パイプラインと同じPython
panorama_script_extra_args =  ; スクリプト固有の引数調整用
```

この場合 `PanoramaSFMStage` はスクリプト経路を自動選択する。

## 実装が仮定しているCLIインターフェース（要検証）

`pipeline/stages/sfm/colmap_panorama.py` は以下のフラグ名を仮定している。
**COLMAP 4.1の実物を入手したら `-h` で確認し、差異があればstageを修正すること。**

prepare（仮想視点レンダリング + database構築、マッピングはスキップ）:

```
colmap panorama_sfm
  --image_path <frames/>        ; ERPフレーム（cubemapではない）
  --database_path <panorama/database.db>
  --output_path <panorama/images/>
  --num_virtual_views 8
  --virtual_view_fov 90
  --camera_model OPENCV
  --use_gpu 1
  --skip_mapping 1
```

map（global）:

```
colmap view_graph_calibrator --database_path <db>   ; run_view_graph_calibrator=True時
colmap global_mapper --database_path <db> --image_path <views> --output_path <sparse>
```

map（incremental）:

```
colmap mapper --database_path <db> --image_path <views> --output_path <sparse>
```

確認用コマンド（入手後にログを残す）:

```powershell
colmap help
colmap panorama_sfm -h
colmap global_mapper -h
colmap view_graph_calibrator -h
```

## 設計メモ

- 入力は必ず `frames/` のERP画像。既存の `cubemap/` 分割画像を入力にすると
  「同一360°フレーム由来の視点をRigとして扱う」関係が失われるため、
  `PanoramaSFMStage` は `frames_dir` が空の場合エラーで停止する。
- 出力は他のSfM stageと同じ `output/<run>/sparse/`（COLMAP sparse形式）に揃え、
  `ctx.sparse_dir` を設定する。下流（3DGSトレーナー）は方式の差を意識しない。
- 中間生成物（仮想視点画像・database）は `output/<run>/panorama/` に置く。
  `keep_intermediate = False` でマッピング後に仮想視点画像を削除できる。
