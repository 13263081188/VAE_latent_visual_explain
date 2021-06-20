import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO,StringIO
from torchvision.transforms.functional import resize, to_tensor, normalize, to_pil_image
import sys
import os
sys.path.append('//app//latent_cam')
sys.path.append(os.getcwd())
import latent_cam
from latent_cam import cams
from latent_cam.utils import overlay_mask
import time
import torch
import importlib

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
    print("X_____________________",x)
    print(type(x),"_+++++++++++++++++++++++")
    print(dict(x.named_modules()))
    submodule_dict = dict(x.named_modules())
    conv = []
    for i in submodule_dict:
        if 'encoder' in i:
            conv.append(i)
    return conv

# @st.cache
def main():
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

        uploaded_file_py = st.sidebar.file_uploader("Upload model.py", type=['py'])
        uploaded_file_pth = st.sidebar.file_uploader("Upload model_parameters", type=['pth'])
        # Choose your own image
        uploaded_file_img = st.sidebar.file_uploader("Upload image", type=['png', 'jpeg', 'jpg'])

        print("testing____________________________________",uploaded_file_py == None, uploaded_file_pth == None)
        if uploaded_file_py:
            #上传模型文件，
            model_stringio = StringIO(uploaded_file_py.getvalue().decode("utf-8"))
            #以string流读入文件内容
            st.write(uploaded_file_py.name)
            model_str = model_stringio.read()
            st.write(model_str)
            #创建临时目录存放model
            if "temp_models" not in os.listdir():
                os.mkdir("temp_models")

            #将模型写入指定文件夹
            write_model = open("temp_models//"+uploaded_file_py.name,'w')
            write_model.write(model_str)

            #加载初始化文件
            init_model = open("temp_models//__init__.py",'w')
            init_model.write("from ."+uploaded_file_py.name[:-3]+" import *")

            #关闭文件
            write_model.close()
            init_model.close()
            import temp_models
            #必须使用reload函数，否则import的内容不会变化
            importlib.reload(temp_models)
        st.sidebar.title("Setup")

        latent_size = st.sidebar.text_input(label="latent_size")

        if uploaded_file_pth and latent_size:
            print("INT_______________________________________________________")
            # 加载模型参数，将tensor变为cpu类型
            checkpoint = torch.load(BytesIO(uploaded_file_pth.read()), map_location='cpu')
            print(checkpoint['state_dict']['fc1.bias'].shape,"FULLCONECT_____________________________________")
            # print(checkpoint['state_dict'].keys())
            layer = checkpoint['state_dict'].keys()

            # Model selection
            # 获取文件对象
            print("LATENT_______SIZE",latent_size)
            latent_size = int(latent_size)
            model = temp_models.__dict__[uploaded_file_py.name[:-3]](latent_size)
            # 解析模型层名
            con_layer = get_encoder_con_layer(model)
            #选择待解释的隐变量的位置
            l_num = list(range(int(latent_size)))
            latent_pos = st.sidebar.selectbox("latent_pos", l_num)
            print("load_state_dict___________________________________++++++++++++++++++++++++++++")
            #给模型加载参数
            model.load_state_dict(checkpoint['state_dict'])
            target_layer = con_layer[-1]
            print("con_layer",con_layer)
            #选择待解释层
            target_layer = st.sidebar.selectbox("select_layer", con_layer)

        if uploaded_file_img is not None:
            # this kind of format is jpeg,<class 'PIL.JpegImagePlugin.JpegImageFile'>
            # print("Image.open(BytesIO(uploaded_file.read()), mode='r')",type(Image.open(BytesIO(uploaded_file.read()), mode='r')))
            # this kind of format is <class 'bytes'>
            # print("uploaded_file.read()",
            #       type(uploaded_file.read()))
            img = Image.open(BytesIO(uploaded_file_img.read()), mode='r').convert('RGB')
            # img = resize(img, (28, 28)).convert('RGB')
            # this kind of format is  <class 'PIL.Image.Image'>
            zz.image(img, use_column_width=True)

        for i in range(len(CAM_METHODS)):
            if cols[i].form_submit_button("解释图计算 V" + CAM_METHODS[i]):
                cam_method = CAM_METHODS[i]
                st.write(cam_method)
                if cam_method is not None:
                   #提取解释方法
                   cam_extractor = cams.__dict__[cam_method](
                         model.eval(),
                         target_layer=target_layer if len(target_layer) > 0 else None
                     )
                if uploaded_file_py is None or uploaded_file_img is None or uploaded_file_pth is None:
                    st.sidebar.error("Please upload an image first")
                else:
                    with st.spinner('Analyzing...'):
                        #预处理图片，归一化图片
                        img_tensor = normalize(to_tensor(resize(img, (28, 28))), [0.485],
                                           [0.229])
                        #将模型作为输入传给模型得到输出
                        scores = model(img_tensor.unsqueeze(0))
                        mu = scores[1]
                        logvar = scores[2]
                        #计算某个隐变量相对于某特征层的解释图
                        activation_map = cam_extractor(int(latent_pos), model.reparameterize(mu,logvar))
                        print(activation_map.size())
                        x, y, z = cols[i].beta_columns(3)
                        # Plot the raw heatmap
                        fig, ax = plt.subplots()
                        ax.imshow(activation_map.numpy())
                        ax.axis('off')
                        x.image(img, use_column_width=True)
                        im = Image.fromarray(activation_map.numpy()).convert('RGB')
                        y.pyplot(fig)
                        # Overlayed CAM
                        fig, ax = plt.subplots()
                        # about mode https://blog.csdn.net/u013066730/article/details/102832597
                        # F represent that grey and pixel value-float 32
                        result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                        ax.imshow(result)
                        ax.axis('off')
                        z.pyplot(fig)
        #展示使用各种可视化解释方法产生的解释图
        if all.form_submit_button("comupte_all"):
            list1 = all.beta_columns(len(CAM_METHODS))
            for i in range(len(CAM_METHODS)):
                cam_method = CAM_METHODS[i]
                if cam_method is not None:
                    cam_extractor = cams.__dict__[cam_method](
                        model.eval(),
                        target_layer=target_layer if len(target_layer) > 0 else None
                    )
                start = time.time()
                list1[i].text("V"+CAM_METHODS[i][:14])
                if uploaded_file_py is None or uploaded_file_img is None or uploaded_file_pth is None:
                    st.sidebar.error("Please upload an image first")
                else:
                    with st.spinner('Analyzing...'):
                        #预处理图片，对图片进行归一化处理
                        img_tensor = normalize(to_tensor(resize(img, (28, 28))), [0.485],
                                           [0.229])
                        scores = model(img_tensor.unsqueeze(0))
                        mu = scores[1]
                        logvar = scores[2]
                        # Retrieve the CAM
                        latent_z = model.reparameterize(mu, logvar)
                        activation_map = cam_extractor(int(latent_pos), latent_z)
                        # Plot the raw heatmap
                        fig, ax = plt.subplots()
                        ax.imshow(activation_map.numpy())
                        ax.axis('off')
                        list1[i].pyplot(fig)
                        im = Image.fromarray(activation_map.numpy())

                        # Overlayed CAM
                        fig, ax = plt.subplots()
                        # about mode https://blog.csdn.net/u013066730/article/details/102832597
                        # F represent that grey and pixel value-float 32
                        result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                        ax.imshow(result)
                        ax.axis('off')
                        list1[i].pyplot(fig)
                        list1[i].text("time:"+str(round((time.time()-start)/1000,3))+'s')

        #对所有隐变量生成的解释图进行融合
        all_integrate = st.form("all_latent_integrate")
        all_integrate.header("all_latent_integrate")
        if all_integrate.form_submit_button("all_latent_integrate"):
            list1 = all_integrate.beta_columns(len(CAM_METHODS))
            for i in range(len(CAM_METHODS)):
                cam_method = CAM_METHODS[i]
                if cam_method is not None:
                    cam_extractor = cams.__dict__[cam_method](
                        model.eval(),
                        target_layer=target_layer if len(target_layer) > 0 else None
                    )
                start = time.time()
                list1[i].text("V" + CAM_METHODS[i][:14])
                if uploaded_file_py is None or uploaded_file_img is None or uploaded_file_pth is None:
                    st.sidebar.error("Please upload an image first")
                else:
                    with st.spinner('Analyzing...'):
                        # 预处理图片，对图片进行归一化处理
                        img_tensor = normalize(to_tensor(resize(img, (28, 28))), [0.485],
                                               [0.229])
                        scores = model(img_tensor.unsqueeze(0))
                        mu = scores[1]
                        logvar = scores[2]
                        # Retrieve the CAM
                        latent_z = model.reparameterize(mu, logvar)
                        activation_map = torch.zeros(7,7)
                        for latent_pos_ in range(latent_size):
                            temp = cam_extractor(int(latent_pos_), latent_z)
                            activation_map += torch.where(torch.isnan(temp), torch.full_like(temp, 0), temp)

                        # Plot the raw heatmap
                        fig, ax = plt.subplots()
                        ax.imshow(activation_map.numpy())
                        ax.axis('off')
                        list1[i].pyplot(fig)
                        im = Image.fromarray(activation_map.numpy())

                        # Overlayed CAM
                        fig, ax = plt.subplots()
                        # about mode https://blog.csdn.net/u013066730/article/details/102832597
                        # F represent that grey and pixel value-float 32
                        result = overlay_mask(img, to_pil_image(activation_map, mode='F'), alpha=0.5)
                        ax.imshow(result)
                        ax.axis('off')
                        list1[i].pyplot(fig)
                        list1[i].text("time:" + str(round((time.time() - start) / 1000, 3)) + 's')
        st.write("delete_the_all_module______________________________________________")
        # 删除所有的模型文件
        # for i in os.listdir("temp_models"):
        #     os.remove(i)
            # print(model(32))
if __name__ == '__main__':
    main()

