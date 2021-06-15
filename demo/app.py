# Copyright (C) 2020-2021, François-Guillaume Fernandez.
# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO,StringIO
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image
import sys
import os
sys.path.append('//app//torch-cam')
sys.path.append('..//torch-cam')

print(sys.path)
from latent_cam import cams
from latent_cam.utils import overlay_mask
import time
import torch


# import convae
# import vae_models
import temp_models
# import ConvVAE

CAM_METHODS = ["GradCAM", "GradCAMpp", "SmoothGradCAMpp", "ScoreCAM", "SSCAM", "ISCAM", "XGradCAM"]
# TV_MODELS = ["resnet18", "resnet50", "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large"]
# ENCODER = ["Conv","Resnet18"]
# TARGET= {"ConvVAE":"encoder.2"}
# VAE_MODELS = ["VAE","BETA-VAE"]

beta_= [0.5,1,2]
# LABEL_MAP = requests.get(
#     "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
# ).json()
def get_encoder_con_layer(x):
    submodule_dict = dict(x.named_modules())
    conv = []
    for i in submodule_dict:
        if 'encoder' in i:
            conv.append(i)
    return conv
# @st.cache
def main():
        print(sys.path)
        # Wide mode
        st.set_page_config(layout="wide", page_title="Variation-auto-encoder interpret")
        # Designing the interface
        # For newline
        st.write('\n')
        test = st.beta_columns(3)
        zz = test[0].form("input_image")
        zz.write("INPUT_IMAGE")
        zz.form_submit_button("dont'touch")


        cols = [st.form(str(i)) for i in range(len(CAM_METHODS))]
        st.write('\n')
        for i in range(len(CAM_METHODS)):
            cols[i].write(CAM_METHODS[i])
        all = st.form("ALL")
        all.header("COMPUTE_ALL")

        all.write()
        # Sidebar
        # File selection

        st.sidebar.title("Input selection")
        # Disabling warning
        st.set_option('deprecation.showfileUploaderEncoding', False)
        # Choose your own image
        uploaded_file_img = st.sidebar.file_uploader("Upload image", type=['png', 'jpeg', 'jpg'])
        if uploaded_file_img is not None:
            #this kind of format is jpeg,<class 'PIL.JpegImagePlugin.JpegImageFile'>
            # print("Image.open(BytesIO(uploaded_file.read()), mode='r')",type(Image.open(BytesIO(uploaded_file.read()), mode='r')))
            #this kind of format is <class 'bytes'>
            # print("uploaded_file.read()",
            #       type(uploaded_file.read()))
            img = Image.open(BytesIO(uploaded_file_img.read()), mode='r').convert('RGB')
            # img = resize(img, (28, 28)).convert('RGB')
            #this kind of format is  <class 'PIL.Image.Image'>
            zz.image(img, use_column_width=True)

        uploaded_file_pth = st.sidebar.file_uploader("Upload model_parameters", type=['pth'])
        if uploaded_file_pth is not None:
            #加载模型参数，将tensor变为cpu类型
            checkpoint = torch.load(BytesIO(uploaded_file_pth.read()), map_location='cpu')
            # print(checkpoint['state_dict'].keys())
            layer = checkpoint['state_dict'].keys()
            #获取VAE中隐变量的个数
            latent_size = len(checkpoint['state_dict']['fc1.bias'])
            print("latent_size______________________________________________________________",latent_size)
            print("CHECKCHECKCHECKCHECK___________________CHECK________CHECK___________________")

        uploaded_file_py = st.sidebar.file_uploader("Upload model.py", type=['py'])
        if uploaded_file_py is not None:
            #上传模型文件，
            model_stringio = StringIO(uploaded_file_py.getvalue().decode("utf-8"))
            #以string流读入文件内容
            st.write(uploaded_file_py.name)
            model_str = model_stringio.read()
            st.write(model_str)
            #将模型写入指定文件夹
            write_model = open("temp_models\\"+uploaded_file_py.name,'w')
            #加载初始化文件
            init_model = open("temp_models\\__init__.py",'a')
            write_model.write(model_str)

            init_model.write("\nfrom ."+uploaded_file_py.name[:-3]+" import *")
            #关闭文件
            write_model.close()
            init_model.close()

        print("______________________________aaaaaa")
        import temp_models
        print(os.listdir('temp_models'))
        print(temp_models.__dict__.keys())
        st.write(temp_models.__dict__)
        for i in temp_models.__dict__:
            print(i)

        print("+++++++++++++++++++++")
        #获取文件对象
        model = temp_models.__dict__[uploaded_file_py.name[:-3]]


        con_layer = get_encoder_con_layer(model)
        # Model selection
        st.sidebar.title("Setup")
        l_num = list(range(latent_size))
        latent_pos = st.sidebar.selectbox("latent_pos", l_num)
        print("load_state_dict___________________________________++++++++++++++++++++++++++++")
        model.load_state_dict(checkpoint['state_dict'])

        target_layer = con_layer[-1]
        print("con_layer",con_layer)
        target_layer = st.sidebar.selectbox("select_layer", con_layer)
        for i in range(len(CAM_METHODS)):
            # cols[i + 1].form_submit_button("COMPUTE " + CAM_METHODS[i])
            # for i in range(1,4):
            # st.write
            # print("OUT________")
            if cols[i].form_submit_button("解释图计算 V" + CAM_METHODS[i]):

                print("co-------------------------------")
                st.title("COOO"+CAM_METHODS[i])
                cam_method = CAM_METHODS[i]
                st.write(cam_method)
                print("int")
                if cam_method is not None:
                   cam_extractor = cams.__dict__[cam_method](
                         model.eval(),
                         target_layer=target_layer if len(target_layer) > 0 else None
                     )
                print("out_int")
                # st.write(cam_method)
                st.balloons()
                if uploaded_file is None:
                    st.sidebar.error("Please upload an image first")
                else:
                    # st.balloons()
                    with st.spinner('Analyzing...'):
                        # Preprocess image
                        img_tensor = normalize(to_tensor(resize(img, (28, 28))), [0.485],
                                           [0.229])
                        print("PIL")
                        print(img_tensor.shape)
                      # Forward the image to the model
                        scores = model(img_tensor.unsqueeze(0))
                        print(model)
                        mu = scores[1]
                        logvar = scores[2]
                        print("mu,logvae", mu.shape, logvar.shape)
                       # Select the target class
                        # if class_selection == "Predicted class (argmax)":
                #         #     class_idx = out.squeeze(0).argmax().item()
                #         # else:
                #         #     class_idx = LABEL_MAP.index(class_selection.rpartition(" - ")[-1])
                #
                       # Retrieve the CAM
                       #  print(model)
                        activation_map = cam_extractor(int(latent_pos), model.reparameterize(mu,logvar))
                        # print("avtivation_map", type(activation_map))
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
                                #         # im > PIL.Image.Image
                        im = Image.fromarray(activation_map.numpy()).convert('RGB')
                        print(type(im))
                        y.pyplot(fig)
                        # Overlayed CAM
                        fig, ax = plt.subplots()
                #
                #         # about mode https://blog.csdn.net/u013066730/article/details/102832597
                #         # F represent that grey and pixel value-float 32
                        result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                        ax.imshow(result)
                #         #                     im = Image.fromarray(result).convert('RGB')
                        print("result", type(result))
                        ax.axis('off')
                #         # cols_3.write("1")
                #         # cols_2.pyplot(fig)
                #         z.image(img,use_column_width=True)
                        z.pyplot(fig)




        if all.form_submit_button("comupte_all"):
            list1 = all.beta_columns(len(CAM_METHODS))
            print("allll")
            for i in range(len(CAM_METHODS)):
                # cols[i + 1].form_submit_button("COMPUTE " + CAM_METHODS[i])
                # for i in range(1,4):
                # if cols[i].form_submit_button("COMPUTE " + CAM_METHODS[i]):
                #     st.balloons()
                cam_method = CAM_METHODS[i]
                # st.write(cam_method)
                if cam_method is not None:
                    cam_extractor = cams.__dict__[cam_method](
                        model.eval(),
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
                        img_tensor = normalize(to_tensor(resize(img, (28, 28))), [0.485],
                                           [0.229])
                        print("img_tensor", img_tensor.shape)
                        scores = model(img_tensor.unsqueeze(0))

                        mu = scores[1]
                        logvar = scores[2]
                        # Select the target class
                        # if class_selection == "Predicted class (argmax)":
                        #         #     class_idx = out.squeeze(0).argmax().item()
                        #         # else:
                        #         #     class_idx = LABEL_MAP.index(class_selection.rpartition(" - ")[-1])
                        #
                        print("mu,logvae",mu.shape,logvar.shape)
                        print(model)
                        # Retrieve the CAM
                        latent_z = model.reparameterize(mu, logvar)
                        print(latent_z.shape)
                        activation_map = cam_extractor(int(latent_pos), latent_z)
                        print("avtivation_map", type(activation_map), activation_map.shape)


                        # Forward the image to the model
                        # out = model(img_tensor.unsqueeze(0))
                        # Select the target class
                        # if class_selection == "Predicted class (argmax)":
                        #     class_idx = out.squeeze(0).argmax().item()
                        # else:
                        #     class_idx = LABEL_MAP.index(class_selection.rpartition(" - ")[-1])
                        # Retrieve the CAM
                        # activation_map = cam_extractor(int(latent_pos), out)
                        # print("avtivation_map", type(activation_map))
                        # print(activation_map.size())
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
        st.write("delete_the_all_module______________________________________________")
        # 删除所有的模型文件
        for i in os.listdir("temp_models"):
            os.remove(i)
            # print(model(32))
if __name__ == '__main__':
    main()

