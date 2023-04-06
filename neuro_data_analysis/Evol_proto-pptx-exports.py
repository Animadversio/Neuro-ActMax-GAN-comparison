import os.path
import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
from scipy.stats import sem
from pptx.util import Inches
from pptx import Presentation  # , SlidePart
#%%
attr_root = r"E:\Network_Data_Sync\BigGAN_FeatAttribution"
protosumdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
meta_df = pd.read_csv(join(tabdir, "meta_stats.csv"), index_col=False)
#%% Create masks for summarizing the prototype images
from neuro_data_analysis.neural_data_lib import load_img_resp_pairs, load_neural_data, get_expstr, \
    extract_evol_activation_array
_, BFEStats = load_neural_data()
#%%
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoSummary"
act_S_col = []
for Expi in tqdm(range(1, len(BFEStats)+1)):  # 66 is not good
    try:
        explabel = get_expstr(BFEStats, Expi)
    except:
        continue
    if BFEStats[Expi-1]["evol"] is None:
        continue
    S = BFEStats[Expi-1]
    resp_arr0, bsl_arr0, gen_arr0, resp_vec0, bsl_vec0, gen_vec0 = \
        extract_evol_activation_array(BFEStats[Expi-1], thread=0)
    resp_arr1, bsl_arr1, gen_arr1, resp_vec1, bsl_vec1, gen_vec1 = \
        extract_evol_activation_array(BFEStats[Expi-1], thread=1)
    # imgfps_arr0, resp_arr0, bsl_arr0, gen_arr0 = \
    #     load_img_resp_pairs(BFEStats, Expi, "Evol", thread=0, output_fmt="arr")
    # imgfps_arr1, resp_arr1, bsl_arr1, gen_arr1 = \
    #     load_img_resp_pairs(BFEStats, Expi, "Evol", thread=1, output_fmt="arr")
    # if the lAST BLOCK has < 10 images, in either thread, then remove it
    if len(resp_arr0[-1]) < 10:
        resp_arr0 = resp_arr0[:-1]
        bsl_arr0 = bsl_arr0[:-1]
        gen_arr0 = gen_arr0[:-1]
    if len(resp_arr1[-1]) < 10:
        resp_arr1 = resp_arr1[:-1]
        bsl_arr1 = bsl_arr1[:-1]
        gen_arr1 = gen_arr1[:-1]

    #%% max block mean response for each thread and their std. dev.
    blck_m_0 = np.array([arr.mean() for arr in resp_arr0])  # np.mean(resp_arr0, axis=1)
    blck_m_1 = np.array([arr.mean() for arr in resp_arr1])  # np.mean(resp_arr1, axis=1)

    maxrsp_blkidx_0 = np.argmax(blck_m_0, axis=0)
    maxrsp_blkidx_1 = np.argmax(blck_m_1, axis=0)
    max_blk_resps_0 = resp_arr0[maxrsp_blkidx_0]
    max_blk_resps_1 = resp_arr1[maxrsp_blkidx_1]
    end_blk_resps_0 = resp_arr0[-1]
    end_blk_resps_1 = resp_arr1[-1]
    init_blk_resps_0 = resp_arr0[0]
    init_blk_resps_1 = resp_arr1[0]
    stats = {"Expi": Expi, "ephysFN": BFEStats[Expi-1]["meta"]["ephysFN"],
             "maxrsp_0_mean": max_blk_resps_0.mean(), "maxrsp_0_std": max_blk_resps_0.std(), "maxrsp_0_sem": sem(max_blk_resps_0),
             "maxrsp_1_mean": max_blk_resps_1.mean(), "maxrsp_1_std": max_blk_resps_1.std(), "maxrsp_1_sem": sem(max_blk_resps_1),
             "endrsp_0_mean": end_blk_resps_0.mean(), "endrsp_0_std": end_blk_resps_0.std(), "endrsp_0_sem": sem(end_blk_resps_0),
             "endrsp_1_mean": end_blk_resps_1.mean(), "endrsp_1_std": end_blk_resps_1.std(), "endrsp_1_sem": sem(end_blk_resps_1),
             "initrsp_0_mean": init_blk_resps_0.mean(), "initrsp_0_std": init_blk_resps_0.std(), "initrsp_0_sem": sem(init_blk_resps_0),
             "initrsp_1_mean": init_blk_resps_1.mean(), "initrsp_1_std": init_blk_resps_1.std(), "initrsp_1_sem": sem(init_blk_resps_1),
             }
    act_S_col.append(stats)
#%%
act_df = pd.DataFrame(act_S_col)
#%%
tabdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Stats_tables"
meta_df = pd.read_csv(join(tabdir, "meta_stats.csv"), index_col=False)
meta_df.rename(columns={"Unnamed: 0": "Expi"}, inplace=True)
#%%
# list(meta_df.columns)
meta_act_df = meta_df.merge(act_df, on=["Expi", "ephysFN"], how="left")  #.to_csv(join(tabdir, "meta_stats.csv"), index=False)
meta_act_df.to_csv(join(tabdir, "meta_activation_stats.csv"), index=False)
#%%
meta_act_df = pd.read_csv(join(tabdir, "meta_activation_stats.csv"), index_col=False)
#%%
ppt_file = Presentation()
for Expi in tqdm(range(1, 190+1)):  # 66 is not good
    imgpath = join(protosumdir, f"Exp{Expi}_proto_attr_summary.png")
    if os.path.exists(imgpath):
        # add slide
        slide = ppt_file.slides.add_slide(ppt_file.slide_layouts[6])
        # add image
        slide.shapes.add_picture(imgpath, left=Inches(0.5), top=Inches(0.0), height=Inches(7.5)) #width=Inches(10),

ppt_file.save(join(protosumdir, "proto_attr_summary.pptx"))

#%%
def create_pptx_from_exps(expids):
    ppt_file = Presentation()
    for Expi in tqdm(expids):  # 66 is not good
        imgpath = join(protosumdir, f"Exp{Expi}_proto_attr_summary.png")
        if os.path.exists(imgpath):
            # add slide
            slide = ppt_file.slides.add_slide(ppt_file.slide_layouts[6])
            # add image
            slide.shapes.add_picture(imgpath, left=Inches(0.5), top=Inches(0.0), height=Inches(7.5)) #width=Inches(10),
        else:
            print(f"Exp{Expi} image not found")
    return ppt_file
#%%
succmsk = (meta_act_df.p_maxinit_0 < 0.01) & \
          (meta_act_df.p_maxinit_1 < 0.01)

cmpmsk = (meta_act_df.maxrsp_1_mean - meta_act_df.maxrsp_0_mean).abs() \
       < (meta_act_df.maxrsp_0_sem + meta_act_df.maxrsp_1_sem)

#%%
Expids = meta_act_df[succmsk & cmpmsk].Expi.values
pptx_cmp_bothsucc = create_pptx_from_exps(Expids)
pptx_cmp_bothsucc.save(join(protosumdir, "proto_comparable_success.pptx"))
#%%
pptx_bothsucc = create_pptx_from_exps(meta_act_df[succmsk].Expi.values)
pptx_bothsucc.save(join(protosumdir, "proto_both_success.pptx"))

#%%
imgcmp_root = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp"
imgcmpdir = r"E:\OneDrive - Harvard University\Manuscript_BigGAN\Figures\ProtoImage_cmp\scratch"
def create_pptx_from_exps_plus_cmp(expids):
    ppt_file = Presentation()
    for Expi in tqdm(expids):  # 66 is not good
        # if os.path.exists(imgpath):
        # add slide
        slide = ppt_file.slides.add_slide(ppt_file.slide_layouts[6])
        # add image
        slide.shapes.add_picture(join(protosumdir, f"Exp{Expi}_proto_attr_summary.png"),
                                 left=Inches(0.5), top=Inches(0.0), height=Inches(7.5)) #width=Inches(10),

        slide = ppt_file.slides.add_slide(ppt_file.slide_layouts[6])
        slide.shapes.add_picture(join(imgcmpdir, f"Exp{Expi:02d}_FC_BG_reevol_pix.png"),
                                 left=Inches(0.5), top=Inches(0.0), height=Inches(7.5))  # width=Inches(10),

        slide = ppt_file.slides.add_slide(ppt_file.slide_layouts[6])
        slide.shapes.add_picture(join(imgcmpdir, f"Exp{Expi:02d}_FC_BG_reevol_G.png"),
                                 left=Inches(0.5), top=Inches(0.0), height=Inches(7.5))  # width=Inches(10),

        slide = ppt_file.slides.add_slide(ppt_file.slide_layouts[6])
        slide.shapes.add_picture(join(imgcmpdir, f"Exp{Expi:02d}_FC_BG_maxblk.png"),
                                 left=Inches(0.5), top=Inches(0.0), height=Inches(7.5))  # width=Inches(10),
        # else:
        #     print(f"Exp{Expi} image not found")
    return ppt_file


pptx_file = create_pptx_from_exps_plus_cmp(meta_df.Expi.values)
pptx_file.save(join(imgcmp_root, "proto_attr_summary_plus_cmp.pptx"))

#%%
