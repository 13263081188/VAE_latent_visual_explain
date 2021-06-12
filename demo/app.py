# Copyright (C) 2020-2021, François-Guillaume Fernandez.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image
from torchcam import cams
from torchcam.utils import overlay_mask
import time
import torch
import convae
import os
CAM_METHODS = ["CAM", "GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM"]
# TV_MODELS = ["resnet18", "resnet50", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]
ENCODER = ["Conv"]
TARGET= {"ConvVAE":"encoder.2"}

VAE_MODELS = ["VAE"]
beta_= [0.5,1,2]
# LABEL_MAP = requests.get(
#     "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# ).json()


# @st.cache
def main():
        # Wide mode
        print("++++__________________________Dictt")
        print('test_in_____________________________________________________________________________')
        import os
        print(os.getcwd())
        # da_ = torch.load('..\\..\\pth\\checkpoint.pth', map_location='cpu')
        # layer_name = (da_['state_dict']).keys()
        print()
        print("test_out")
        print("test_in_out")
        st.set_page_config(layout="wide",page_title="变分自动编码器可视化解释器")
        # Designing the interface
        st.title("变分自动编码器可视化解释系统")
        # For newline
        st.write('\n')
        test = st.beta_columns(3)
        zz = test[0].form("input image")
        # zz.form_submit_button("dont'touch")

        # cam_ for i in range(1000)
        cols = [st.form(str(i)) for i in range(len(CAM_METHODS))]
        # cols[0].write("Input image")
        # zz.form_submit_button("DONT'touch me")
        st.write('\n')
        for i in range(len(CAM_METHODS)):
            cols[i].write(CAM_METHODS[i])
            cols[i].form_submit_button("解释图计算 V" + CAM_METHODS[i])
            # x,y,z = cols[i + 1].beta_columns(3)
            # x.write("1")
            # y.write("2")
            # z.write("3")
        all = st.form("ALL")
        all.header("COMPUTE_ALL")
        # Sidebar
        # File selection
        st.sidebar.title("Input selection")
        # Disabling warning
        st.set_option('deprecation.showfileUploaderEncoding', False)
        # Choose your own image
        uploaded_file = st.sidebar.file_uploader("Upload image", type=['png', 'jpeg', 'jpg'])
        # print(uploaded_file)
        if uploaded_file is not None:
            # zz.write("Aa Aa Aa")
            # st.balloons()
            #this kind of format is jpeg,<class 'PIL.JpegImagePlugin.JpegImageFile'>
            # print("Image.open(BytesIO(uploaded_file.read()), mode='r')",type(Image.open(BytesIO(uploaded_file.read()), mode='r')))
            #this kind of format is <class 'bytes'>
            # print("uploaded_file.read()",
            #       type(uploaded_file.read()))
            img = Image.open(BytesIO(uploaded_file.read()), mode='r').convert('RGB')
            #this kind of format is  <class 'PIL.Image.Image'>
            # print("IMG",type(img))
            zz.image(img, use_column_width=True)
        if zz.form_submit_button("Input Image"):
            # if uploaded_file is not None:
            # st.balloons()
            pass
        # print("++++__________________________Dictt")
        # Model selection
        st.sidebar.title("Setup")
        encoder = st.sidebar.selectbox("encoder_model", ENCODER)
        default_layer = ""
        # if tv_model is not None:
        #     with st.spinner('Loading model...'):
        #         model = models.__dict__[tv_model](pretrained=True).eval()
        #     default_layer = cams.utils.locate_candidate_layer(model, (3, 224, 224))
        l_num = list(range(32))
        latent_pos = st.sidebar.selectbox("latent_pos", l_num)
        vae_model = st.sidebar.selectbox("VAE_structure", VAE_MODELS)
        mode_name = encoder+vae_model
        # if vae_model == "beta_vae":
        #     beta = st.sidebar.selectbox("beta_value", [0.5,1,2])
        #     mode_name += str(beta)
        # print()
        print("camsDictt")
        # print()
        # for qwe in cams.__dict__:
        #     print(qwe, cams.__dict__[qwe])
        # print(cams.__dict__)
        model = convae.ConvVAE(3).eval()
        # print("model_up")
        import os
        # print(os.listdir())
        checkpoint = torch.load('..\\pth\\checkpoint.pth', map_location='cpu')
        # print("load_up")
        model.load_state_dict(checkpoint['state_dict'])
        # print("load_already")
        # if vae_model is not None and latent_num is not None:
        #     with st.spinner('Loading model...'):
        #         model = models.__dict__[tv_model](pretrained=True).eval()
        # default_layer = cams.utils.locate_candidate_layer(model, (3, 224, 224))
        # default_layer = ""
        target_layer = "encoder.2"
        # print("heeeeeee")
        # cam_method = st.sidebar.selectbox("CAM method", CAM_METHODS)
        # st.write(cam_method)
        # if cam_method is not None:
        #     cam_extractor = cams.__dict__[cam_method](
        #         model,
        #         target_layer=target_layer if len(target_layer) > 0 else None
        #     )
        # st.write(cam_method)

        # class_choices = [f"{idx + 1} - {class_name}" for idx, class_name in enumerate(LABEL_MAP)]
        # class_selection = st.sidebar.selectbox("Class selection", ["Predicted class (argmax)"] + class_choices)
        # print("heree")
        for i in range(len(CAM_METHODS)):
            # cols[i + 1].form_submit_button("COMPUTE " + CAM_METHODS[i])
            # for i in range(1,4):
            print("OUT________")
            if cols[i].form_submit_button("解释图计算 V" + CAM_METHODS[i]):
                print("COOO")
                cam_method = CAM_METHODS[i]
                # st.write(cam_method)
                print("int")
                if cam_method is not None:
                    cam_extractor = cams.__dict__[cam_method](
                        model,
                        target_layer=target_layer if len(target_layer) > 0 else None
                    )
                print("out_int")
                # st.write(cam_method)
                st.balloons()
                if uploaded_file is None:
                    st.sidebar.error("Please upload an image first")
                else:
                    st.balloons()
                    with st.spinner('Analyzing...'):
                        # Preprocess image
                        img_tensor = normalize(to_tensor(resize(img, (28, 28))), [0.485],
                                           [0.229])
                        print("PIL")
                        # Forward the image to the model
                        scores = model(img_tensor.unsqueeze(0))
                        mu = scores[1]
                        logvar = scores[2]
                        # Select the target class
                        # if class_selection == "Predicted class (argmax)":
                        #     class_idx = out.squeeze(0).argmax().item()
                        # else:
                        #     class_idx = LABEL_MAP.index(class_selection.rpartition(" - ")[-1])

                        # Retrieve the CAM
                        activation_map = cam_extractor(int(latent_pos), model.reparameterize(mu,logvar))
                        print("avtivation_map", type(activation_map))
                        print(activation_map.size())
                        x, y, z = cols[i].beta_columns(3)
                        # Plot the raw heatmap
                        fig, ax = plt.subplots()
                        ax.imshow(activation_map.numpy())
                        ax.axis('off')
                        # cols_1,cols_2,cols_3 = cols[i].beta_columns(3)
                        x.image(img, use_column_width=True)
                        # y.image(img,use_column_width=True)
                        # cols_1.write('1')
                        # cols_2.write("1")

                        # im > PIL.Image.Image
                        im = Image.fromarray(activation_map.numpy()).convert('RGB')
                        print(type(im))
                        y.pyplot(fig)
                        # Overlayed CAM
                        fig, ax = plt.subplots()

                        # about mode https://blog.csdn.net/u013066730/article/details/102832597
                        # F represent that grey and pixel value-float 32
                        result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                        ax.imshow(result)
                        #                     im = Image.fromarray(result).convert('RGB')
                        print("result", type(result))
                        ax.axis('off')
                        # cols_3.write("1")
                        # cols_2.pyplot(fig)
                        # z.image(img,use_column_width=True)
                        z.pyplot(fig)
        if all.form_submit_button("comupte_all"):
            list1 = all.beta_columns(len(CAM_METHODS))
            for i in range(len(CAM_METHODS)):
                # cols[i + 1].form_submit_button("COMPUTE " + CAM_METHODS[i])
                # for i in range(1,4):
                # if cols[i].form_submit_button("COMPUTE " + CAM_METHODS[i]):
                #     st.balloons()
                cam_method = CAM_METHODS[i]
                # st.write(cam_method)
                if cam_method is not None:
                    cam_extractor = cams.__dict__[cam_method](
                        model,
                        target_layer=target_layer if len(target_layer) > 0 else None
                    )
                start = time.time()
                list1[i].text("V"+CAM_METHODS[i][:14])
                if uploaded_file is None:
                    st.sidebar.error("Please upload an image first")
                else:
                    # st.balloons()
                    with st.spinner('Analyzing...'):
                        # Preprocess image
                        img_tensor = normalize(to_tensor(resize(img, (224, 224))), [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])
                        # Forward the image to the model
                        out = model(img_tensor.unsqueeze(0))
                        # Select the target class
                        # if class_selection == "Predicted class (argmax)":
                        #     class_idx = out.squeeze(0).argmax().item()
                        # else:
                        #     class_idx = LABEL_MAP.index(class_selection.rpartition(" - ")[-1])
                        # Retrieve the CAM
                        activation_map = cam_extractor(int(latent_pos), out)
                        print("avtivation_map", type(activation_map))
                        print(activation_map.size())
                        # x, y, z = cols[i].beta_columns(3)
                        # Plot the raw heatmap
                        fig, ax = plt.subplots()
                        ax.imshow(activation_map.numpy())
                        ax.axis('off')
                        # cols_1,cols_2,cols_3 = cols[i].beta_columns(3)
                        # x.image(img, use_column_width=True)
                        # y.image(img,use_column_width=True)
                        # cols_1.write('1')
                        # cols_2.write("1")
                        list1[i].pyplot(fig)
                        # im > PIL.Image.Image
                        im = Image.fromarray(activation_map.numpy())
                        print(type(im))
                        # y.image(im, use_column_width=True)
                        # Overlayed CAM
                        fig, ax = plt.subplots()

                        # about mode https://blog.csdn.net/u013066730/article/details/102832597
                        # F represent that grey and pixel value-float 32
                        result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                        ax.imshow(result)
                        # im = Image.fromarray(result).convert('RGB')
                        print("result", type(result))
                        ax.axis('off')
                        # cols_3.write("1")
                        # cols_2.pyplot(fig)
                        list1[i].pyplot(fig)
                        list1[i].text("time:"+str(round((time.time()-start)/1000,3))+'s')
                        # z.image(img,use_column_width=True)
                        # z.image(im, use_column_width=True)
if __name__ == '__main__':
    main()
