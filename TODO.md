# TODO
1. 把latent_dim开到比如32或64，两个模型都重新训练，然后做同样的reconstruction对比可视化
2. vqvae loss playaround 
z_e散点加codebook位置的叠加图。
跑四组实验，每组训练完画同一种图：

完整loss（你现在的baseline）
去掉codebook loss，只留 recon + commitment
去掉commitment loss，只留 recon + codebook
两个都去掉，只留 recon

四张图放一起对比，重点看z_e的散点和红色星星之间的位置关系——是紧密重叠还是彼此分离。
另外再加一个定量指标会更清楚：每个epoch算一下所有z_e到它最近的codebook条目的平均距离，四组实验画在同一张曲线图里。这个距离就是quantization error，直接反映codebook和encoder之间配合得好不好。