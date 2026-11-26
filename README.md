# 緑藻鞭毛の流体相互作用シミュレーション（d/L・粘性率依存性）

Volvox / Pleodorina の鞭毛間の流体相互作用が、鞭毛間隔 d と鞭毛長 L の比 d/L、
および粘性率 η によってどう変わるかを調べる簡易モデル。
平らな no-slip 壁 (z=0) から x 方向に一列 (間隔 d, N 本) に鞭毛が生え、x–z 面内で拍動する。

単位: µm, s, pN, mPa·s（内部では η×1e-3 pN·s/µm²）。依存: numpy / scipy / matplotlib（mp4 出力には ffmpeg、無ければ gif）。

## ファイル

| file | 内容 |
|---|---|
| `model.py` | 正則化 Stokeslet + Blake 鏡像カーネル、拍動波形、ε 較正、モーター較正、時間発展 `run()`、解析 `analyze()`、`rft_check()`、`check_kernel()` |
| `simulate.py` | 1 条件の実行。要約 (summary.txt)・時系列・位相差分布・流れ場・アニメーションを出力 |
| `sweep.py` | d/L × η のスイープ（multiprocessing）と集計グラフ |

## モデル

* **流体**: Stokes 流。各鞭毛を N_seg 点の正則化 Stokeslet（Cortez 2001, blob ε）で表し、壁は Blake (1971) の鏡像系
  （鏡像 Stokeslet + Stokes doublet + source doublet）。鏡像項は |R|→√(|R|²+ε²) で正則化（Ainley et al. 2008 の簡略版）。
  `python model.py` で壁面速度 0 と Faxén/Lorentz の壁近傍移動度 (−9/16, −9/8) を確認できる。
* **ε の較正**: 半径 a=0.125 µm の棒の抵抗（自由空間, 横向き並進 F=4πηLU/(ln(L/a)+0.5)）を再現するよう ε を自動決定
  （`--eps` で直接指定も可）。このため N_seg を下げても力はほぼ変わらない（N_seg=10〜30 で有効打の力の差 0.3%）。
* **波形** (Sleigh / Brennen & Winet 型の簡略版。接線角 ψ(s,φ)、壁法線から測り +x が正):
  * 有効打 (位相の割合 `p_eff`=0.35): ほぼ真っ直ぐな棒が −θ0 → +θ0 に回転（振り下ろし）
  * 回復打: 根元で生じた屈曲が先端へ伝播。基部側は −θ0、先端側は +θ0 のまま壁際を引き戻される
* **拍動モード** `--beat_mode`
  * `fixed`: dφ_i/dt = 2π `f_fixed`（運動学固定。力 ∝ η）
  * `load`: dφ_i/dt = ω0 (1 − T_i/T_stall)。T_i = Σ_k F_k·∂X_k/∂φ_i は φ_i に共役な一般化抵抗トルク
    （散逸 = T_i φ̇_i）。他の鞭毛の流れを含むので相互作用する。各時刻で φ̇ と力を連立一次方程式として同時に解く。
    ω0 = 2π `f_free`（無負荷 50 Hz）。`T_stall` 未指定なら「単独鞭毛・水中で平均 `f_water`=30 Hz」となるよう自動較正。
* **壁モード** `--wall_mode`
  * `anchored`: 壁静止。推力 = 鞭毛が流体に及ぼす x 方向力（= 基部が壁に伝える力）
  * `swimming`: 壁と基部が速度 U で並進。鞭毛は周囲流体との相対速度 ∂X/∂φ φ̇ + U e_x で力を評価し、
    ζ_body η U + Σ F_x = 0（ζ_body = 6πR, `--R_body`）から毎ステップ U を解く（φ̇ と連立）。
* **交差の扱い**: 全鞭毛が同一平面で拍動するため、位相がずれた隣同士はおよそ d < 1.8L で幾何学的に交差する。
  異なる鞭毛間の相互作用には大きめの blob `eps_inter`（既定 1 µm）を使い発散を防ぐ
  （0.5〜2 µm で結果の変化は ±3% 程度、y 方向に 1 µm ずらす `--y_stagger 1` とほぼ同等）。

## 出力される指標

* `f_i`: 有効打の間だけの x 方向力の時間平均（各鞭毛）、`F_total = Σ f_i`、全周期平均 ⟨ΣFx⟩
* `ε_sim = F_total / (N f_single)`（f_single は同条件 N=1）
* `R_sync`: Kuramoto 秩序パラメータ。位相は単独振動子の一様回転位相 θ(φ)=2π∫dφ/φ̇₁ / T₁ に変換してから評価
  （load モードでは φ̇ が位相で大きく変わるため。一定の時間遅れ ↔ 一定の Δθ）
* 隣接位相差 Δθ = θ_{i+1} − θ_i の分布、円平均、coherence |⟨e^{iΔθ}⟩|、
  pair-locking（各ペアごとの時間方向の |⟨e^{iΔθ}⟩_t| の平均）。
  Δθ<0 は +x（有効打方向）へ進む波 = symplectic、Δθ>0 は antiplectic。
* swimming モードでは遊泳速度 U

## 使い方

```bash
python model.py                                     # カーネル検証 + RFT チェック
python simulate.py --rft_check                      # 同上（パラメータ指定可）
python simulate.py --N 2 --beat_mode fixed          # まず N=2 で動作確認
python simulate.py --N 10 --beat_mode load --species volvox --n_cycles 40
python simulate.py --N 10 --beat_mode load --wall_mode swimming --species pleodorina
python simulate.py --N 10 --beat_mode fixed --init_phase metachronal --dphi_mc -0.785
python sweep.py --beat_mode load --n_cycles 40      # d/L 0.3–1.5 × η {0.84, 2.5, 8.61}
python sweep.py --fast                              # N_seg=12, 40 step/周期
python sweep.py --replot sweep_results/load_anchored_N10
```

全パラメータは `model.Params` にまとまっており、すべて `--名前 値` で上書きできる（`python simulate.py -h`）。
主なもの: `--d --L --dL --eta --N --N_seg --dt --steps_per_cycle --n_cycles --transient --f_fixed --f_free
--f_water --T_stall --R_body --zeta_body --init_phase {random,inphase,metachronal} --dphi_mc --seed
--theta0 --p_eff --eps --eps_inter --y_stagger`。

## Windows で動かす

1. Python 3.10 以上を入れる（python.org のインストーラで「Add python.exe to PATH」にチェック）。
2. このフォルダで `py -m pip install -r requirements.txt`。
3. 実行は `py model.py`、`cd studies` してから `py p7_stability_species.py` など（Linux の `python` を `py` に読み替える）。

- 図の日本語は Yu Gothic / Meiryo / MS Gothic のどれかで表示される（`studies/plot_style.py`）。
- 2 次元カーペットの格子和の表は初回に計算して `.kernel_cache/` に保存する。1 条件あたり数分かかり、全部で 1 GB 程度になる。2 回目からは再利用される。
- 並列計算は `multiprocessing.Pool(4)`。各スクリプトは `if __name__ == "__main__":` で守られているので Windows でもそのまま動く。
- 実測データを使う検証（`studies/p*_*.py`）は、`studies/private/` に `sim_compare_data.csv` と `colony.json` が必要。このフォルダは git の管理外。

## 注意（簡略化）

* 群体表面の曲率・2 次元配列・鞭毛の弾性は考えていない（波形は運動学的に与え、位相の速さだけが流体負荷で変わる）。
* swimming モードの「一様な相対流」は、壁の no-slip と厳密には両立しない近似（仕様どおり）。
* ω0・T_stall・波形パラメータは仮の値。力の絶対値（水中・単独で有効打平均 ~60–80 pN）は振幅 θ0 と周波数に強く依存する。

## d/L と相互作用の強さ（`dl_interaction.py`）

`fixed` モード（運動学固定・同一周波数）で、隣接鞭毛の位相差 Δφ を固定したメタクロナル列（Δφ=0 は同位相）の
ε_sim（端 2 本ずつを除いた内側の鞭毛）を d/L × Δφ の格子で計算する。また、2 本の鞭毛間の流体力学的結合係数
κ = ⟨|Γ12|⟩/⟨Γ11⟩ も計算する。1 条件 1 周期で済むので速い（全体で約 1 分、`--fast` なら 15 秒）。
このモードでは力が η に比例するので、ε_sim と κ は η に依存しない。

```bash
python dl_interaction.py            # → dl_results/dl_interaction.png, .json
```
