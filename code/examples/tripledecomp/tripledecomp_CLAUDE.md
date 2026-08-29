# code/examples/tripledecomp/ 実験規約

三重分解（u = ū + ũ(φ) + u″）と LSUN を組み合わせた実験群。
リポジトリルートの CLAUDE.md の規約（不変量・用語・.mlx 禁止）をすべて継承する。

## この実験群の目的

円柱周り流れ（Kármán 渦列）を対象に、次を検証する：

1. **前提（データ特性）**：対象流れはコヒーレント成分が支配する清浄なリミットサイクルであり、
   位相平均残差はビン幅の 2 乗で減衰する離散化誤差であって乱れ u″ ではない。
2. **主張2（係数力学のユニタリ性）**：コヒーレント成分 ũ(φ) の LSUN 係数列に piDMD を
   適用すると、固有値が単位円上に乗る（リミットサイクル＝中立安定な回転、の座標表現）。
   制約なし同定でも既に単位円上に乗ること（＝制約がデータと整合していること）が要点。

**主張1（基点場の質）は 2026-08-29 に取り下げた。** 手持ちの VORTALL（Re=100）は層流・完全周期で
u″ が実質ゼロ（残差比が nBins に対し指数 −1.94 で減衰し続け、頭打ちにならない）。
比較対象が「コヒーレント成分」対「ビン幅誤差」になってしまい、指標を変えても検証にならない。
乱流後流データが入手できるまで再開しない。

## 実行・テスト方法（このディレクトリでの標準手順）

```
cd code/examples/tripledecomp
matlab -licmode onlinelicensing -batch "setup; results = runtests('tripledecompTestCase.m'); assertSuccess(results)"
matlab -licmode onlinelicensing -batch "setup; main_limitcycle_residual"
matlab -licmode onlinelicensing -batch "setup; main_coeffdyn_unitcircle"
```

- すべての main_* は再実行可能（冪等）にする：乱数は rng(0) 固定、途中生成物は results/ に置く。
- 長時間実験には進捗表示と checkpoint 保存を入れる（parfeval 利用時は ../pidmd/main_pidmdparfeval.mlx の流儀を参照）。

## ファイル構成と役割

- `setup.m`：パス設定（../pidmd, ../tddmd への addpath を含む。../pidmd/setup.m に倣う）
- `fcn_phaseavg.m`：位相同定と位相平均基点場 μ(φ) の推定
  - 入力：スナップショット行列 Y (空間 × 時間)、位相ビン数 nBins
  - 位相同定はリフト量等の断面信号の Hilbert 変換で行う
  - 出力：muPhase (空間 × nBins)、各スナップショットの位相インデックス
- `fcn_divfreeproj.m`：発散フリー射影（Leray 射影）。基点場への物理制約
- `fcn_exportvalues.m`：数値を results/values.json へ書き出し（jsonencode 使用）
- `main_limitcycle_residual.m`：前提（データ特性）の検証。LSUN 学習を伴わないので数分で終わる
- `main_coeffdyn_unitcircle.m`：主張2 の実験本体
- `main_basefield_compare.m`：取り下げた主張1 の実験本体。**休止**。
  乱れ u″ を持つデータが来るまで実行しない（ファイル先頭に DORMANT の注記あり）
- `fcn_lsuntrain2d.m` / `fcn_lsuncoefs2d.m`：2-D LSUN の学習と係数取り出し。
  学習可能パラメータは double へキャストすること（既定の single では不変量が 1e-6 までしか成り立たない）
- `fcn_specwavenumbers.m` / `fcn_specdiv.m`：スペクトル微分の波数と発散。
  偶数長軸の Nyquist 波数は 0 にする（さもないと射影後がエルミートにならず real() で壊れる）
- `tripledecompTestCase.m`：matlab.unittest テスト

## テストに含めるべき項目（tripledecompTestCase.m）

1. ルート規約の不変量 3 点（ユニタリ性・Parseval・完全再構成）を、この実験で構成する
   LSUN インスタンスに対して検証
2. fcn_phaseavg：合成データ（既知位相の正弦波動場）で位相平均が真値を復元すること
3. fcn_divfreeproj：射影後の場の離散発散ノルムが閾値以下、かつ射影の冪等性（P(P(u)) = P(u)）
4. 残差の基点場直交性：位相ビンごとの残差平均がほぼゼロ（‖mean(r|φ)‖ < tol）

## values.json のスキーマ（論文リポジトリとのインターフェース。勝手に変えない）

`fcn_exportvalues.m` が実験ごとに `results/values_<exp>.json` を書き、
それらをマージした `results/values.json` を再生成する。

```json
{
  "exp": "coeffdyn",
  "date": "YYYY-MM-DD",
  "config": {"nBins": 16, "K": 4, "stride": [4,4], "overlap": [3,3], "nModesPod": 10},
  "metrics": {
    "maxUnitCircleDeviation": 0.0,
    "medianUnitCircleDeviation": 0.0,
    "maxUnitCircleDeviationUnitary": 0.0,
    "relFitErrorExact": 0.0,
    "relFitErrorUnitary": 0.0,
    "numBins": 16,
    "numChans": 4,
    "numPodModes": 10
  }
}
```

- **`exp` と `metrics` のキー名は論文側のマクロ名を決める**（`\<exp><MetricKey>`、
  例：`exp="coeffdyn"` × `relFitErrorExact` → `\coeffdynRelFitErrorExact`）。
  **キー名に数字を入れない**。LaTeX の制御綴りは英字のみなので、sync がエラーで止まる。
  論文リポジトリの `sync_results.sh` が機械的に変換するので、キー名を変えると
  本文のマクロ名が変わる。片方だけ変えない。
- 本文が必要とする設定値（位相ビン数・K）は config だけでなく metrics にも出す。
  sync が変換するのは metrics のみのため。

- energyConcentration = 上位 K 係数エネルギー / 全係数エネルギー（ブロック平均）
- 図は results/figures/*.pdf に exportgraphics で出力（フォント埋め込み、'ContentType','vector'）
- 図のファイル名は fig_<実験名>_<内容>.pdf（例：fig_basefield_energyconc.pdf）

## 実験設計上の注意

- 基点場を引いた残差 r(t) = y(t) − μ(φ(t)) に対してのみ LSUN を学習する
  （基点場・基底場の分離学習。基点場は学習せず推定する）。
- 比較条件は「基点場の推定法」のみを変え、LSUN の構成（stride, overlap, K, 学習条件）は
  完全に揃える。乱数シードも共通。
- 主張2 では piDMD のユニタリ制約（Procrustes 解）を使う。../pidmd/fcn_pidmdvialsun.m を再利用し、
  固有値の複素平面プロット（単位円との距離のヒストグラム付き）を出力する。
- データは ../pidmd の円柱流れデータ取得方法に従う。データファイルは results/ にも
  リポジトリにもコミットしない（.gitignore 済みであることを確認）。
