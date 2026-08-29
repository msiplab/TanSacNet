# TanSacNet プロジェクト規約（リポジトリルート）

TanSacNet は Locally-Structured Unitary Network (LSUN) による接空間適応制御ネットワークの研究プロジェクト。
MATLAB が主体（R2024b、Deep Learning Toolbox 等に依存）。PyTorch 版は `code/appendix/torch_tansacnet/`。

## 絶対に守る不変量（これを壊す変更・近似は提案しないこと）

1. **ユニタリ性**：解析辞書 D_θ は常に D_θ * D_θ' = I を満たす。
   検証基準：`norm(D*D' - eye(size(D,1)), 'fro') < 1e-10`
2. **Parseval（エネルギー保存）**：任意の入力 y に対し `abs(norm(y)^2 - norm(D'*y)^2) < 1e-8 * norm(y)^2`
3. **完全再構成**：K = M（全係数保持）のとき `norm(Dsyn(Dana(y)) - y) < 1e-8 * norm(y)`
4. **no-DC-leakage 構造**：第0チャンネルの回転行列 W_0 の第1行・第1列の固定構造
   （W_0' = blkdiag(1, Wbar_0')）を変更しない。DC 成分が AC チャンネルへ漏れないことの構造的保証。

上記 1–3 は対応する unittest（後述）が存在する。コードを変更したら必ずテストを実行して自己検証すること。

## 用語規約（論文とコードで統一。混同しやすいので注意）

- **基点場（base-point field）** = μ：局所データ分布のアンサンブル平均。接空間の基点。
  コード変数名は `muBase`。「バイアス」「DC」と呼ばない。
- **基底場（basis field）** = D_θ：位置ごとのユニタリ基底（フィルタカーネル）。LSUN の学習対象。
- **DC 成分**：定数ベクトル方向（1/sqrt(M)）への射影のみを指す。基点場と同義ではない
  （基点場は一般にエッジ等の任意構造を持ち得る）。
- **三重分解**：u = ū + ũ(φ) + u″（時間平均 + コヒーレント成分 + ランダム乱れ）。
  - ū：時間平均。学習不要（時間平均 + 発散フリー射影で推定）。
  - ũ(φ)：位相平均から時間平均を引いたコヒーレント成分。**LSUN-C の主対象**
    （リミットサイクル＝滑らかな低次元多様体。接空間解釈が厳密に成立）。
  - u″：乱れ。LSUN-T（局所 KLT 解釈）の対象。Reynolds 応力の構造化表現に使う。
- **LSUN 係数**：x = D_θ' * (y − μ)。残差（ゆらぎ）の接空間座標。

## ファイル形式の規則

- **バイナリ .mlx は編集しない**。diff・レビュー不能なため参照のみ。
- **プレーンテキスト版ライブスクリプト（.m 形式）は編集可**。MATLAB R2024a 以降の
  「Save as → Plain Text Live Code file (*.m)」で保存される形式で、`%[text]`・`%[control]`・
  `%[output:...]` などのマークアップを含む通常の .m ファイル。テキストなので diff が取れる。
  編集時はマークアップ行の構文を壊さないこと（本文は `%[text]` 行、コードは素の MATLAB 行）。
- 新規開発はプレーン .m（`%%` セクション区切り、またはプレーンテキスト版ライブスクリプト）で行う。
  公開時に必要なら人間が .mlx へエクスポートする。
- 既存のバイナリ .mlx の処理フローを把握する必要がある場合は、テキスト化した .m
  （export("foo.mlx", "foo.m") の出力）を人間に依頼すること。

## テスト・実行方法（ヘッドレス）

- 全体テスト：`matlab -licmode onlinelicensing -batch "results = runtests('code/mytest.m'); assertSuccess(results)"`
- 個別テスト：`matlab -licmode onlinelicensing -batch "results = runtests('code/lsunXXXTestCase.m'); assertSuccess(results)"`
- テストは matlab.unittest 流儀（既存の code/lsun*TestCase.m に倣う）で書く。

## ディレクトリ構成の要点

- `code/`：MATLAB 本体（カスタムレイヤ群、テストケース群）
- `code/examples/lsun/`：LSUN 基本例（Burgers 等）
- `code/examples/pidmd/`：円柱流れ + LSUN 経由 piDMD（fcn_pidmdvialsun.m, fcn_cylinderplot.m）
- `code/examples/tddmd/`：時間遅れ DMD（Burgers・Lorenz、fcn_lsun_train.m）
- `code/examples/tripledecomp/`：三重分解 × LSUN 実験（新設。専用 CLAUDE.md あり）
- `code/appendix/torch_tansacnet/`：PyTorch 実装

## 車輪の再発明をしない（既存資産の場所）

- piDMD 本体：`code/examples/pidmd/fcn_pidmdvialsun.m`
- 円柱データ可視化：`code/examples/pidmd/fcn_cylinderplot.m`
- LSUN 学習ループ：`code/examples/tddmd/fcn_lsun_train.m`
- 時間発展 DMD：`code/examples/pidmd/fcn_timeevoldmd.m`, `fcn_timeevoldmdwA.m`

## 研究上の主張と実験の対応（原稿は別リポジトリ・非公開）

- 前提「対象流れはコヒーレント成分が支配する清浄なリミットサイクルである
  （＝乱れ u″ が無い）」→ `code/examples/tripledecomp/main_limitcycle_residual.m`
- 主張2「コヒーレント成分の LSUN 係数力学は単位円上の固有値を持つ（ユニタリ piDMD と整合）」
  → `code/examples/tripledecomp/main_coeffdyn_unitcircle.m`
- 主張1「位相平均基点場は単純時間平均より LSUN 係数のエネルギー集中を改善する」は
  **取り下げ（2026-08-29）**。実験本体 `main_basefield_compare.m` は休止状態で残置。
  理由：手持ちの円柱流れデータ（VORTALL, Re=100）は層流・完全周期で u″ が実質ゼロのため、
  この主張を検証できない。乱流後流データが入手できるまで再開しない。
- 数値・図は results/values.json と results/figures/*.pdf 経由でのみ論文リポジトリへ渡す。
  本リポジトリに論文原稿（.tex）を置かない（公開リポジトリのため）。
