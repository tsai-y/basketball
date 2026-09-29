# Research references

Verified 2026-09-27 against the linked publisher or Computer Vision Foundation records. Model and data claims below refer to the papers; project implementation status is stated separately. Do not copy the placeholder metrics in `TOPIC 1.pdf` into experimental results.

## Basketball-51

- **Title:** Basketball-51: A Video Dataset for Activity Recognition in the Basketball Game
- **Authors:** Sarbagya Ratna Shakya, Chaoyang Zhang, Zhaoxian Zhou
- **Year / venue:** 2021; *Computer Science & Information Technology (CS & IT)*, pp. 141–153. The [publisher page](https://aircconline.com/csit/abstract/v11n7/csit110712.html) lists volume 11, number 7, May 2021; the [paper](https://csitcp.org/paper/11/117csit12.pdf) prints this venue and pages.
- **DOI:** [10.5121/csit.2021.110712](https://doi.org/10.5121/csit.2021.110712), printed on the paper's first page.
- **Method / dataset:** 10,311 approximately six-second clips from 51 NBA broadcasts. Eight labels distinguish shot type and make/miss; clips include the ball near the rim and later outcome frames. Section 3.2 defines suffix `0` as miss and `1` as make.
- **Role:** Dataset definition, class mapping, and original full-video basketball classification setting.
- **Implementation status:** Local clips inspected; this project will implement a separately specified binary RGB baseline.
- **Differences:** New duplicate/conflict cleaning and grouped split, binary labels, and modified C3D-style model. These are not a reproduction of the paper's experimental protocol or reported metrics.

## C3D

- **Title:** Learning Spatiotemporal Features with 3D Convolutional Networks
- **Authors:** Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, Manohar Paluri
- **Year / venue:** 2015; IEEE International Conference on Computer Vision (ICCV), pp. 4489–4497, per the [Computer Vision Foundation record](https://openaccess.thecvf.com/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html).
- **Official paper:** [CVF PDF](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf). **DOI:** TODO: verify against publisher record; no DOI is asserted here.
- **Method:** Section 3.3 and Figure 3 show eight `3×3×3` convolutions, five max-pooling layers, two 4096-unit fully connected layers, and classification output. Pool 1 uses temporal kernel/stride 1; subsequent pools reduce time by 2. Section 3.3 specifies 16-frame training crops from resized `128×171` frames; its video descriptor uses continuous 16-frame clips with 8-frame overlap, averages fc6 activations, then L2-normalizes. Section 3.1's non-overlapping clips describe a separate five-convolution architecture search. See [C3D reproduction specification](c3d_reproduction_spec.md) for page-specific evidence and the separate Sports-1M ten-clip prediction protocol.
- **Official implementation:** The [author-released C3D repository](https://github.com/facebookarchive/C3D) provides a [Sports-1M feature extractor definition](https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_feature_extraction/prototxt/c3d_sport1m_feature_extractor_video.prototxt) and [UCF101 fine-tuning definition](https://github.com/facebookarchive/C3D/blob/master/C3D-v1.0/examples/c3d_finetuning/c3d_ucf101_finetuning_train.prototxt). These confirm ReLU positions, 0.5 dropout after fc6/fc7, and convolution padding 1; these are official implementation details, not explicitly drawn in the paper.
- **Role:** Primary architecture and clip-processing reference for a future Basketball-51 adaptation.
- **Implementation status:** Completed project run is a **modified C3D-style** binary RGB baseline; reference-faithful C3D has only an audit/specification and has not been implemented. See `docs/baseline_protocol.md` and `docs/c3d_reproduction_spec.md`.
- **Differences:** Smaller channel widths and classifier, BatchNorm3d after convolutions, 320×240 Basketball-51 input padded/resized to a smaller square, one sampled 16-frame clip per video, binary head, and project-specific optimizer/protocol. It is **not** exact C3D reproduction or use of the paper's pretrained weights.

## I3D

- **Title:** Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
- **Authors:** João Carreira, Andrew Zisserman
- **Year / venue:** 2017; IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 6299–6308, per the [Computer Vision Foundation record](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html).
- **Official paper:** [CVF PDF](https://openaccess.thecvf.com/content_cvpr_2017/papers/Carreira_Quo_Vadis_Action_CVPR_2017_paper.pdf). **DOI:** TODO: verify against publisher record.
- **Method:** Two-stream Inflated 3D ConvNet (I3D) expands 2D convolution/pooling kernels to 3D and studies pretraining on Kinetics.
- **Role:** Reference for a possible later I3D baseline and spatiotemporal features.
- **Implementation status:** Reference only. No I3D implementation, training, or results in this round.
- **Differences:** Not applicable yet.

## Professor's topic note

- **Local source:** `TOPIC 1.pdf`, one-page Chinese research topic note. It sketches possible later YOLO tracking, MixNet/I3D, RL frame selection, SpinalNet and strategy planning.
- **Status:** Research direction only; no cited experiment or implemented architecture in this project. Strings such as `XXXXNet` and `XX.X%` are placeholders, **not measured results**. Bibliographic publication details: TODO: verify if this note is to be cited as a formal publication.
