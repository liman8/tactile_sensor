#!/bin/bash
# Demo za prezentaciju (MuJoCo). Pokreće se iz bilo kog foldera:
#     bash prezentacija/demo.sh <izbor>
#
#   1   ručno: senzor nad viljuškom; klizači u panelu "Control" (press, x, y, yaw ...),
#       taktilna slika i klasični detektor dole levo; Page Down / Page Up = drugi predmet, Home = početak
#   2   pretraga, start 25: licem nadole, prvi pritisak na DRŠCI (dvosmislenost, pa razrešenje)  <- glavni demo
#   3   pretraga, start 18: licem nadole, prvi pritisak na zupcima
#   4   pretraga, start 10: licem nagore, prvi pritisak na vratu
#   5   pretraga, nasumična poza (F9 u prozoru daje novu)
#   6   više predmeta, model A, sa visinom dodira: predmet je nepoznat pretrazi (verovatnoće predmeta u tekstu,
#       senka najverovatnijeg u proceni pune poze);
#       Page Down / Page Up = sledeći / prethodni predmet (viljuška, Fork.stl, nož, kašika)
#   7   model B na viljušci Fork.stl, koju nikad nije video u treningu (kako je planirano: obično je nazove ForkMilimeter)
#   8   isto, ali sa visinom dodira i poravnatom CAD hipotezom (naknadno): Fork.stl prepozna i nađe joj pozu
#   t   provera bez prozora (isti tasteri F6 F6 F8 F9 F7), piše prezentacija/selftest_<vreme>.png
#
# Tasteri u pretrazi: F6 jedan pritisak, F7 do kraja, F8 briše verovanje, F9 nova viljuška.
# Na laptopu F6-F9 često traže Fn. Pretraga: bočni meniji MuJoCo-a su sakriveni, Tab / Shift+Tab ih prikazuju;
# gore desno je pogled odozgo (siva = prava viljuška, zelena = procena) sa brojevima. T = providna scena,
# levi taster + vuci = okretanje kamere, desni taster + vuci = pomeranje, točkić = zum.
# Ne koristiti Ctrl + vuci: to primenjuje silu na izabrano telo.
set -e
cd "$(dirname "$0")/.."
PY=python
[ -x .venv/bin/python ] && PY=.venv/bin/python
NET=${NET:-results_centre/mini_robust/model_100k_seed0.pt}     # trenutni kandidat za konačni model
[ -f "$NET" ] || { echo "nema modela $NET"; exit 1; }
case "${1:-}" in
  1) exec $PY live_viewer.py ;;
  2) exec env GELSIGHT_SENSOR=mini $PY search_viewer.py --net "$NET" --start 25 ;;
  3) exec env GELSIGHT_SENSOR=mini $PY search_viewer.py --net "$NET" --start 18 ;;
  4) exec env GELSIGHT_SENSOR=mini $PY search_viewer.py --net "$NET" --start 10 ;;
  5) exec env GELSIGHT_SENSOR=mini $PY search_viewer.py --net "$NET" --seed "${2:-7}" ;;
  6) [ -f results_multi/multi_all/model_seed0.pt ] || { echo "nema modela results_multi/multi_all/model_seed0.pt"; exit 1; }
     exec env GELSIGHT_SENSOR=mini $PY search_viewer.py --model A --object "${2:-knife_stl}" --seed 3 --height ;;
  7) [ -f results_multi/multi_holdout/model_seed0.pt ] || { echo "nema modela results_multi/multi_holdout/model_seed0.pt"; exit 1; }
     exec env GELSIGHT_SENSOR=mini $PY search_viewer.py --model B --object fork2_stl --seed 3 ;;
  8) [ -f results_multi/multi_holdout/model_seed0.pt ] || { echo "nema modela results_multi/multi_holdout/model_seed0.pt"; exit 1; }
     exec env GELSIGHT_SENSOR=mini $PY search_viewer.py --model B --object fork2_stl --seed 3 --height --cad-align ;;
  t) out=prezentacija/selftest_$(date +%H%M%S).png
     MUJOCO_GL=egl GELSIGHT_SENSOR=mini $PY search_viewer.py --net "$NET" --selftest "$out" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac
