# ROAM 运行与验证命令

以下命令针对当前机器：Ubuntu 22.04、ROS 2 Humble，ROAM 在 `/home/wzw/Desktop/dog_nav/roam`，工作空间在 `/home/wzw/Desktop/dog_nav/ros2_ws`。统一使用 `roam/.venv` 的 Python 3.10 环境，并开启 `--system-site-packages` 访问 ROS 2 Humble 的系统包。

每段命令均包含工作目录，可单独复制。API 密钥从 `~/.codex/auth.json` 读取，不写入仓库。Docker 和系统软件安装使用 `sudo`，Python/ROS 命令不要用 `sudo`。`MPLBACKEND=Agg` 表示保存图像而不弹窗。

## 安装：从零配置

以下步骤假设 ROAM 仓库已位于上述路径，系统为 Ubuntu 22.04。先确认 NVIDIA 驱动、Docker + NVIDIA Container Toolkit、ROS 2 Humble 和 `uv` 已安装；缺少的系统组件需要先安装，Python 虚拟环境无法代替它们：

```bash
nvidia-smi
docker --version
nvidia-ctk --version
uv --version
test -f /opt/ros/humble/setup.bash && echo 'ROS 2 Humble installed'
```

若缺少 Docker/GPU 容器支持，需先安装系统 Docker 和 NVIDIA Container Toolkit；这些是系统级组件，不由 `uv` 管理。Docker 能看到 GPU 的验证命令：

```bash
sudo docker run --rm --gpus all ubuntu nvidia-smi
```

安装构建和 ROS 2 系统依赖：

```bash
sudo apt update
sudo apt install -y python3-colcon-common-extensions python3-rosdep \
  ros-humble-grid-map-msgs ros-humble-tf-transformations
sudo rosdep init 2>/dev/null || true
rosdep update
```

ROAM 和 ROS 共用新建的 Python 3.10 环境 `roam/.venv`。`--system-site-packages` 让它能加载 ROS Humble 的 Python 扩展；先安装匹配本机 CUDA 12.8 的 PyTorch，再安装 ROAM、Depth Anything V2、高程地图和 SOCKS 依赖：

```bash
cd /home/wzw/Desktop/dog_nav/roam
uv venv --python /usr/bin/python3 --system-site-packages
UV_HTTP_TIMEOUT=180 uv pip install --python .venv/bin/python \
  'torch==2.11.0+cu128' torchvision \
  --index-url https://download.pytorch.org/whl/cu128 && \
uv pip install --python .venv/bin/python -e './vlm_nav[depth]' \
  'numpy<1.24' 'cupy-cuda12x<14' simple-parsing shapely scipy ruamel.yaml ros2-numpy socksio
```

`&&` 保证 cu128 PyTorch 安装成功后才继续安装 ROAM 依赖。若下载超时，重试整段；安装完成后用第 0 节检查 `torch.version.cuda`。

uv环境激活：

```bash
. /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
```

高程地图源码单独放在 `/home/wzw/Desktop/dog_nav/elevation_mapping_cupy`。下载支持 ROS 2 语义层的分支，再应用图像投影 TF 方向补丁：

```bash
cd /home/wzw/Desktop/dog_nav
git clone -b pascal-roth/b2w-wild-navigation-semantic-layers \
  https://github.com/leggedrobotics/elevation_mapping_cupy.git
cd elevation_mapping_cupy
git fetch https://github.com/mktk1117/elevation_mapping_cupy.git \
  fix/image-callback-tf-direction
git cherry-pick FETCH_HEAD
```

首次把 ROS 包放进工作空间并构建。高程仓库里有其他 ROS 1 包，这里只链接 ROS 2 融合需要的两个子包：

```bash
mkdir -p /home/wzw/Desktop/dog_nav/ros2_ws/src
cd /home/wzw/Desktop/dog_nav/ros2_ws/src
ln -s /home/wzw/Desktop/dog_nav/roam/vlm_nav .
ln -s /home/wzw/Desktop/dog_nav/roam/vlm_nav_ros .
ln -s /home/wzw/Desktop/dog_nav/elevation_mapping_cupy/elevation_map_msgs .
ln -s /home/wzw/Desktop/dog_nav/elevation_mapping_cupy/elevation_mapping_cupy .

cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
python -m colcon build --symlink-install \
  --packages-select elevation_map_msgs elevation_mapping_cupy vlm_nav vlm_nav_ros \
  --cmake-args -DPython3_EXECUTABLE=/home/wzw/Desktop/dog_nav/roam/.venv/bin/python
source install/setup.bash
```

上述安装、克隆和链接步骤仅首次执行；已经存在的目录或链接无需重复创建。

## 0. 环境检查

```bash
cd /home/wzw/Desktop/dog_nav/roam
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -c 'import rclpy, cv_bridge, torch, vlm_nav, transformers, socksio; print("ROAM/ROS:", torch.__version__, torch.version.cuda, torch.cuda.is_available())'

cd /home/wzw/Desktop/dog_nav/ros2_ws
source install/setup.bash
python -c 'import pydantic, cupy, elevation_mapping_cupy; print("map packages OK", cupy.__version__)'
head -1 install/vlm_nav_ros/lib/vlm_nav_ros/vlm_nav_node
```

最后一行应指向 `/home/wzw/Desktop/dog_nav/roam/.venv/bin/python`。

## 1. SAN 服务：终端 A

单图 VLM + SAN、跳过 VLM 重新分割，以及离线重算语义图需要此服务。看到 `Running on http://127.0.0.1:7860` 后保持终端 A 运行。第 4 节的录制语义图重放不需要 SAN。

```bash
cd /home/wzw/Desktop/dog_nav/roam
sudo ./vlm_nav/bin/run_san_server.sh
```

## 2. 单图生成语义图：终端 B

保持终端 A 运行。每执行一次 `example.py` 都会调用中转站 VLM 和 SAN。先列弹窗版，再列无弹窗版；两者分别运行，不需要都运行。

`hiking_lake.jpg` 弹窗版（关掉四联图后命令退出）：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
export OPENAI_API_KEY="$(jq -r '.OPENAI_API_KEY' ~/.codex/auth.json)"

../.venv/bin/python example.py example_data/hiking_lake.jpg \
  --seg_model api --vlm_model gpt-5.6-terra \
  --vlm_base_url https://app.tool2ai.com/api/codex --save_images
```

`hiking_lake.jpg` 无弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
export OPENAI_API_KEY="$(jq -r '.OPENAI_API_KEY' ~/.codex/auth.json)"
MPLBACKEND=Agg ../.venv/bin/python example.py example_data/hiking_lake.jpg \
  --seg_model api --vlm_model gpt-5.6-terra \
  --vlm_base_url https://app.tool2ai.com/api/codex --save_images
```

`stepping_stones_stream.jpg` 弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
export OPENAI_API_KEY="$(jq -r '.OPENAI_API_KEY' ~/.codex/auth.json)"
../.venv/bin/python example.py example_data/stepping_stones_stream.jpg \
  --seg_model api --vlm_model gpt-5.6-terra \
  --vlm_base_url https://app.tool2ai.com/api/codex --save_images
```

`stepping_stones_stream.jpg` 无弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
export OPENAI_API_KEY="$(jq -r '.OPENAI_API_KEY' ~/.codex/auth.json)"
MPLBACKEND=Agg ../.venv/bin/python example.py example_data/stepping_stones_stream.jpg \
  --seg_model api --vlm_model gpt-5.6-terra \
  --vlm_base_url https://app.tool2ai.com/api/codex --save_images
```

检查结果：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
ls -lh result/hiking_lake/ result/stepping_stones_stream/
```

每个目录应有 `overview.png`、`labels.json`、`vlm_response.json`、`input.png`，以及 `risk`、`steppable`、`preferred_area` 各自的彩色图和 `_gray.png` 数值图。同名图片重跑会覆盖同名结果。

仅重新分割已有标签，不请求 VLM 或使用 API 密钥；终端 A 仍须运行。
弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
../.venv/bin/python example.py example_data/hiking_lake.jpg \
  --labels result/hiking_lake/labels.json --seg_model api --save_images
```

无弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
MPLBACKEND=Agg ../.venv/bin/python example.py example_data/hiking_lake.jpg \
  --labels result/hiking_lake/labels.json --seg_model api --save_images
```

这会覆盖 `result/hiking_lake/` 中的同名图片，不会产生新的 VLM 返回内容。

## 3. 二维图像 A* 规划

读取第 2 节的 `_gray.png`，不再请求 VLM 或 SAN。起终点是 `行,列` 像素坐标。先运行弹窗版（关闭窗口后命令退出）：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
../.venv/bin/python click_plan_demo.py example_data/hiking_lake.jpg \
  --maps_dir result/hiking_lake --no_depth
```

左键设终点，右键改起点；默认起点在图片底部中央。

Hiking Lake 固定目标的无弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
MPLBACKEND=Agg ../.venv/bin/python click_plan_demo.py example_data/hiking_lake.jpg \
  --maps_dir result/hiking_lake --no_depth \
  --goal 245,330 --save result/hiking_lake/plan_no_depth.png
```

`--inflate 5` 仅用于观察禁行区不扩张的演示效果，不代表真实机器人的安全距离。
过溪图片交互弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
../.venv/bin/python click_plan_demo.py example_data/stepping_stones_stream.jpg \
  --maps_dir result/stepping_stones_stream --no_depth
```

过溪图片固定目标弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
../.venv/bin/python click_plan_demo.py example_data/stepping_stones_stream.jpg \
  --maps_dir result/stepping_stones_stream --no_depth --inflate 4 \
  --goal 210,150
```

过溪图片固定目标无弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
MPLBACKEND=Agg ../.venv/bin/python click_plan_demo.py example_data/stepping_stones_stream.jpg \
  --maps_dir result/stepping_stones_stream --no_depth --inflate 4 \
  --goal 210,150 --save result/stepping_stones_stream/plan_no_depth.png
```

启用 Depth Anything V2 相对深度时，去掉 `--no_depth`。首次运行需要从 Hugging Face 下载模型；SOCKS 代理还要求 `roam/.venv` 安装 `socksio`。深度弹窗版（首次会下载模型）：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
../.venv/bin/python click_plan_demo.py example_data/hiking_lake.jpg \
  --maps_dir result/hiking_lake --inflate 4
```

深度无弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/roam/vlm_nav
MPLBACKEND=Agg ../.venv/bin/python click_plan_demo.py example_data/hiking_lake.jpg \
  --maps_dir result/hiking_lake \
  --save result/hiking_lake/plan_with_depth.png \
  --goal 230,297 --inflate 4
```

深度只调整像素代价，不产生真实世界坐标。

## 4. 录制数据离线融合高程地图

`data/seealpsee_trail/clip.npz` 包含点云、位姿和已生成的语义图。此测试直接调用 `elevation_mapping_cupy` 的 GPU 地图核心，无需 ROS 话题、真实传感器、Docker 或 API 密钥。当前源码分支是 `roam_ros2`。

离线高程地图固定目标弹窗版（显示俯视图，点击地图单元格规划；关闭窗口后退出）：

（1）默认：使用录制好的语义结果，不调用 VLM。

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
cd /home/wzw/Desktop/dog_nav/roam/examples/elevation_mapping
python semantic_elevation_map.py --goal=-16,-21
```

同一固定目标的无弹窗版（指定目标并保存）：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash

cd /home/wzw/Desktop/dog_nav/roam/examples/elevation_mapping
MPLBACKEND=Agg python semantic_elevation_map.py --goal=-16,-21 --save map.png
ls -lh map.png
```

目标是 `odom` 地图中的 `x,y`，单位米。负数目标必须写成 `--goal=-10,-3`，不能写成 `--goal -10,-3`。

`Axes3D` 的 Matplotlib 警告不妨碍二维地图生成。

可选：终端 A 保持 SAN 服务运行，使用片段中记录的 VLM 标签，重新对每帧相机图片做 SAN 分割后再融合。这比直接重放录制语义图慢，不需要 API 密钥：

弹窗版（关闭窗口后退出）：

（2）使用录制的 VLM 标签，但重新调用 SAN 分割。

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
cd /home/wzw/Desktop/dog_nav/roam/examples/elevation_mapping
python semantic_elevation_map.py --source segment --seg_model api
```

无弹窗版：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
cd /home/wzw/Desktop/dog_nav/roam/examples/elevation_mapping
MPLBACKEND=Agg python semantic_elevation_map.py --source segment \
  --seg_model api --save map_resegmented.png
```

`--source vlm` 会重新请求 VLM；当前离线示例没有 `--vlm_base_url` 参数，因此这里不提供中转站运行命令。

## 5. ROS 2 构建与启动

当前 `ros2_ws/src` 已链接 ROAM、`elevation_map_msgs` 和 `elevation_mapping_cupy`。修改 Python 源码或安装配置后，用 Python 3.10 环境构建。缺少 `pytest` 的警告只影响测试，不影响构建。

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
python -m colcon build --symlink-install \
  --packages-select elevation_map_msgs elevation_mapping_cupy vlm_nav vlm_nav_ros \
  --cmake-args -DPython3_EXECUTABLE=/home/wzw/Desktop/dog_nav/roam/.venv/bin/python
source install/setup.bash
```

保持终端 A 的 SAN 服务运行。单独启动 ROAM ROS 节点：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
export OPENAI_API_KEY="$(jq -r '.OPENAI_API_KEY' ~/.codex/auth.json)"
ros2 launch vlm_nav_ros vlm_nav_ros.launch.py
```

同时启动 ROAM 和高程地图节点（替代上一条 launch，不要两个一起运行）：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
export OPENAI_API_KEY="$(jq -r '.OPENAI_API_KEY' ~/.codex/auth.json)"
ros2 launch vlm_nav_ros vlm_nav_elevation_mapping.launch.py
```

这两条 ROS 命令仅启动节点。当前配置预期实际话题 `/zed/zed_node/rgb/image_rect_color`、`/zed/zed_node/rgb/camera_info`、`/lidar/points`，以及 `odom`/`base` 等 TF。没有传感器或 ROS bag 发布数据时，**不会得到持续更新的融合地图**；验证融合算法请运行第 4 节。有数据时另开已 source 工作空间的终端检查：

```bash
ros2 node list
ros2 topic list
ros2 topic hz /zed/zed_node/rgb/vlm_nav_sem
ros2 topic echo --once /vlm_nav_node/labels
```

中转站地址、模型和 SAN 模式在 `vlm_nav_ros/config/vlm_nav_cfg.yaml`。
点云话题和 TF 帧在 `vlm_nav_ros/config/elevation_mapping_vlm_nav.yaml`。
连接真实机器人前须按实际话题与帧名修改。

## 7. ROS Humble 的 `np.float` 错误

如果高程地图启动时报 `module 'numpy' has no attribute 'float'`，这是 ROS Humble 的 `transforms3d` 与 NumPy 1.24 以上版本不兼容。将共享虚拟环境降到 NumPy 1.23.5（不需要重编译 ROS 包）：

```bash
cd /home/wzw/Desktop/dog_nav/roam
uv pip install --python .venv/bin/python 'numpy==1.23.5'
source /opt/ros/humble/setup.bash
source .venv/bin/activate
python -c 'import numpy as np; from tf_transformations import quaternion_matrix; print("numpy", np.__version__, "tf_transformations OK")'
```

新环境安装步骤已固定为 `numpy<1.24`，避免再次触发该兼容问题。

## 6. 无传感器 ROS 2 数据回放测试

新增的 `examples/elevation_mapping/replay_ros2.py` 将 `data/seealpsee_trail` 的录制数据发布成 ROS 消息，模拟传感器输入。它使用当前 ROS 时间，不发布 `/clock`，保持 `use_sim_time=false`。无需重新编译回放脚本。

| 话题 | 类型/内容 |
|---|---|
| `/lidar/points` | `sensor_msgs/PointCloud2`，XYZ，单位米，`lidar` 坐标系 |
| `/zed/zed_node/rgb/image_rect_color` | `sensor_msgs/Image`，`rgb8`，320×213 |
| `/zed/zed_node/rgb/camera_info` | `sensor_msgs/CameraInfo`，匹配图像尺寸和录制内参 |
| `/tf` | `odom → base`、`odom → lidar`、`odom → camera_optical_frame` 的录制位姿 |

录制数据只包含约 30 秒的稀疏帧，不等同于原始高频传感器流。RGB 与语义图统一到内参对应的尺寸；相机名称沿用现有话题配置，图像与畸变参数仍来自录制相机。

### 6.1 完整链路：回放传感器 → VLM → SAN → 高程地图

先按第 1 节启动 SAN。另开终端 B，运行第 5 节的联合启动命令：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
export OPENAI_API_KEY="$(jq -r '.OPENAI_API_KEY' ~/.codex/auth.json)"
ros2 launch vlm_nav_ros vlm_nav_elevation_mapping.launch.py
```

等待节点完成初始化后，终端 C 慢速回放一次（约 2 分钟）：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
cd /home/wzw/Desktop/dog_nav/roam/examples/elevation_mapping
python replay_ros2.py --rate 0.25
```

此模式会实际调用中转站 VLM，并由 SAN 重新分割。`--rate 1` 是原速，`--loop` 可重复播放；循环时机器人位姿跳回起点，地图不会自动清空。片段结束后脚本保持运行，按 Ctrl+C 退出。若 VLM 请求过慢导致首轮没有语义图，可降低回放速度后重新运行。

### 6.2 离线融合：直接发布录制语义图

此模式无需 VLM、SAN 或 API 密钥。先停止上述联合 launch，避免两个节点同时发布同一个语义话题。终端 B 只启动高程地图节点：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
ros2 run elevation_mapping_cupy elevation_mapping_node.py --ros-args \
  --params-file /home/wzw/Desktop/dog_nav/elevation_mapping_cupy/elevation_mapping_cupy/config/core/core_param.yaml \
  --params-file /home/wzw/Desktop/dog_nav/roam/vlm_nav_ros/config/elevation_mapping_vlm_nav.yaml
```

终端 C 发布录制传感器数据，同时额外发布 `/zed/zed_node/rgb/vlm_nav_sem`（`32FC3`，通道依次为 risk、steppable、preferred_area）、`/zed/zed_node/rgb/camera_info_sem` 和供 rqt 查看用的 `/zed/zed_node/rgb/vlm_nav_sem_viz`（`bgr8`，从左到右是原图、risk、steppable、preferred_area）：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
cd /home/wzw/Desktop/dog_nav/roam/examples/elevation_mapping
python replay_ros2.py --publish-recorded-semantics --rate 0.5 --loop
```
--publish-recorded-semantics：把数据集中已保存的 risk、`steppable`、`preferred_area` 语义图直接发布到 ROS，供高程图融合；不调用 VLM 或 SAN。
--rate 0.5：以原录制速度的一半回放，约 30 秒的数据需要约 60 秒。
--loop：播完后从头再播，直到按 Ctrl+C 停止。

### 6.3 检查回放结果

终端 D（回放进行中执行）：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
ros2 node list
ros2 topic list -t
ros2 topic hz /lidar/points
```

`ros2 topic hz` 会一直运行；按 Ctrl+C 后可以改查语义图：

```bash
ros2 topic hz /zed/zed_node/rgb/vlm_nav_sem
```

查看 TF：

```bash
ros2 run tf2_ros tf2_echo odom camera_optical_frame
```

查看相机和语义可视化（弹窗，需要安装 `ros-humble-rqt-image-view`）：

```bash
ros2 run rqt_image_view rqt_image_view
```

在界面选择 `/zed/zed_node/rgb/image_rect_color` 或 `/zed/zed_node/rgb/vlm_nav_sem_viz`。直接选 `/zed/zed_node/rgb/vlm_nav_sem` 会看到浮点数值图，rqt 通常只显示灰度。高程地图属于 `grid_map_msgs/GridMap`，可通过 `ros2 topic list -t` 找到实际输出话题；两个 launch 和回放脚本均不自动启动 RViz 或 A* 导航节点。


### 话题说明

| 话题 | 简单说明 |
|---|---|
| `/zed/zed_node/rgb/image_rect_color` | 回放的原始相机图像 |
| `/zed/zed_node/rgb/camera_info` | 相机内参和畸变参数 |
| `/lidar/points` | 激光雷达三维点云 |
| `/tf` | `odom`、相机、雷达和机器人之间的位姿变换 |
| `/zed/zed_node/rgb/vlm_nav_sem` | 三通道语义数值图：`risk`、`steppable`、`preferred_area`。通常显示为灰色 |
| `/zed/zed_node/rgb/vlm_nav_sem_viz` | 便于人查看的彩色语义可视化图 |
| `/zed/zed_node/rgb/camera_info_sem` | 与语义图对应的相机参数 |


### 6.4 拖拽查看融合后的立体高程图

高程地图节点和数据回放运行时，在新终端启动独立查看器。等待地图话题有数据后，窗口会显示实时更新的立体曲面；将鼠标移到曲面所在的坐标轴区域，左键拖拽旋转，滚轮缩放。关闭窗口即可退出。请勿设置 `MPLBACKEND=Agg`。

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
cd /home/wzw/Desktop/dog_nav/roam/examples/elevation_mapping
python visualize_gridmap_3d.py --color-layer risk
```

上面以高程作为曲面高度、风险作为颜色（绿低风险、红高风险）。只看高程着色时运行：

```bash
cd /home/wzw/Desktop/dog_nav/ros2_ws
source /opt/ros/humble/setup.bash
source /home/wzw/Desktop/dog_nav/roam/.venv/bin/activate
source install/setup.bash
cd /home/wzw/Desktop/dog_nav/roam/examples/elevation_mapping
python visualize_gridmap_3d.py
```

默认订阅 `/elevation_mapping_node/elevation_map_raw`。若输出话题不同，先用 `ros2 topic list -t` 查找 `grid_map_msgs/msg/GridMap`，再给查看器传 `--topic /实际话题`。缺测高程单元不会绘制；录制结束后显示最后一张地图。`--stride 2` 是默认抽样比例，若旋转卡顿可以用 `--stride 4`。
