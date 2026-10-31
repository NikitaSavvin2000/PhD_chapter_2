
ssh nsavvin@192.168.1.67

mkdir -p /Users/nikitasavvin/Desktop/PhD/PhD_chapter_2/export_server && scp -r nsavvin@192.168.1.67:~/exp/PhD_chapter_2/export/prod_npt_setup_17_10_2026_v1 /Users/nikitasavvin/Desktop/PhD/PhD_chapter_2/export_server/


mkdir -p /Users/nikitasavvin/Desktop/PhD/PhD_chapter_2/export_server && scp -r nsavvin@95.165.85.82:~/exp/PhD_chapter_2/export/prod_npt_setup_13_10_2026_v2 /Users/nikitasavvin/Desktop/PhD/PhD_chapter_2/export_server/


ssh nsavvin@95.165.85.82


export PYTHONPATH=$PWD/src:$PYTHONPATH


mkdir -p /Users/nikitasavvin/Desktop/PhD/3_phd_chapter/PhD_chapter_2/export/export_server && scp -r nsavvin@192.168.1.67:~/exp/PhD_chapter_2/export/prod_npt_setup_17_10_2026_v1 /Users/nikitasavvin/Desktop/PhD/3_phd_chapter/PhD_chapter_2/export/export_server

mkdir -p /Users/nikitasavvin/Desktop/PhD/PhD_chapter_2/export/export_server && scp -r nsavvin@95.165.85.82:~/exp/PhD_chapter_2/export/prod_not_setup_26_10_2026_v1 /Users/nikitasavvin/Desktop/PhD/3_phd_chapter/PhD_chapter_2/export/export_server



/exp/PhD_chapter_2/export/prod_not_setup_30_10_2026_v1


scp -r nsavvin@192.168.1.67:~/exp/PhD_chapter_2/export/prod_not_setup_30_10_2026_v1/events.csv /Users/nikitasavvin/Desktop/PhD/3_phd_chapter/PhD_chapter_2/export/export_server