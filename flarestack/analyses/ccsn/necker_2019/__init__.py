# import os
# from flarestack.analyses.ccsn.stasik_2017.shared_ccsn import sn_catalogue_name
#
# ccsn_dir = os.path.abspath(os.path.dirname(__file__))
# ccsn_cat_dir = ccsn_dir + "/catalogues/"
# raw_cat_dir = ccsn_cat_dir + "raw/"
#
# # sn_cats = ["IIn", "IIp", "Ibc"]
# sn_cats = ["IIn"]
#
# sn_times = [100., 300., 1000.]
# sn_times = [300.]
#
# def updated_sn_catalogue_name(sn_type, nearby=True):
#     sn_name = sn_type + "_"
#
#     if nearby:
#         sn_name += "nearby.npy"
#     else:
#         sn_name += "distant.npy"
#
#     return ccsn_cat_dir + sn_name
#
# def sn_time_pdfs(sn_type):
#
#     time_pdfs = []
#
#     for i in sn_times:
#         time_pdfs.append(
#             {
#                 "time_pdf_name": "box",
#                 "pre_window": 0,
#                 "post_window": i
#             }
#         )
#
#     if sn_type == "Ibc":
#         time_pdfs.append(
#             {
#                 "time_pdf_name": "box",
#                 "pre_window": 20,
#                 "post_window": 0
#             }
#         )
#
#     return time_pdfs