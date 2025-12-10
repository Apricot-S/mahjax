# MahJax

JAX で実装した強化学習向けの高速マージャン環境とエージェント実装/UIのセットです。

注意: JAX のインストールは環境( CPU / CUDA / TPU )に依存します。まずご利用環境に合った手順で JAX を入れてください。

## ディレクトリ構成
- `mahjax/no_red_mahjong` — 赤なしルールの環境実装 (現状こちらを使っている)
- `mahjax/red_mahjong` — 赤ありルールの実験的実装 (未完成)
- `mahjax/agents` — データ収集・オフライン/オンライン学習・評価用スクリプト一式
- `mahjax/ui` — FastAPI ベースの Human vs AI Web UI（`/static` にフロント資材）
- `mahjax/_src` — 可視化や内部ユーティリティ（SVG アニメーション等）
- `tests` — 主要モジュールのテスト
- `benchmark` — ベンチマーク用スクリプト
- `fig` — 可視化の出力先（学習スクリプトが書き出します）
- `mahjax/initial_implementation` — pgxのコピー

## セットアップ（最小）
1) 依存関係をインストール
- `pip install -r requirements.txt`


2) 本リポジトリをインストール（開発モード推奨）
- `pip install -e .`

## UI の立ち上げ方（Human vs AI）
FastAPI のアプリファクトリを Uvicorn で起動します。

- 起動: `python -m uvicorn mahjax.ui.app:create_app --factory --reload`
- 既定: `http://127.0.0.1:8000/` にアクセス
- 収録エージェント: ルールベース / ランダム

補足:
- 静的資材は `mahjax/ui/static` にあります。トップページは `/` で配信されます。
- 学習済みモデルとの接続は今後整備予定です。暫定的に `mahjax/ui/agents.py` にエージェントを追加して利用できます。

## エージェント訓練の方法
JAX/Flax ベースのネットワークで、(A) ルールベース自己対戦によるオフラインデータ収集 → (B) オフライン学習 → (C) オンライン学習（REMAX 風）を想定しています。各スクリプトは OmegaConf の CLI 引数に対応しています。

事前に学習用依存関係をインストールしてください:
`pip install flax optax distrax omegaconf wandb tqdm numpy pydantic`

### A. オフラインデータ収集
- 例: `python -m mahjax.agents.collect_offline_data num_envs=64 num_jitted_steps=16 total_timesteps=1000000 save_path=mahjong_offline_data.pkl`
- 出力: `mahjong_offline_data.pkl`（観測・行動・価値ターゲットなど）

### B. オフライン学習（模倣 + 価値回帰）
- 例: `python -m mahjax.agents.offline_train dataset=mahjong_offline_data.pkl batch_size=512 num_epochs=10 num_channels=128 num_blocks=6 sparsity=0.8`
- 出力:
  - 学習済みチェックポイント: `il-offline-training.ckpt`
  - 一局プレイの可視化: `fig/il_agent_animation.svg`

### C. オンライン学習（自己対戦）
- 例: `python -m mahjax.agents.online_train num_envs=64 num_steps=128 total_timesteps=5000000 init_params=il-offline-training.ckpt sparsity=0.8`
- 出力:
  - 学習済みチェックポイント: `mahjong_remax.ckpt`
  - 一局プレイの可視化: `fig/rl_agent_animation.svg`
- ロギング: Weights & Biases を使用します。ネットワークに接続できない場合は `WANDB_MODE=disabled` を環境変数で指定してください。

## テスト
- 例: `pytest -q`

## メモ
- JAX はバージョンやデバイス依存が大きいため、エラー時は公式のインストールガイドに従って再インストールしてください。
- 高速化のため初回実行時に JIT コンパイルが走ります（最初だけ時間がかかります）。
# mahjax
