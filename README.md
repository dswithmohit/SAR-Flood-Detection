SAR

Synthetic Aperture Radar (SAR) is a form of “active sensing” it means that it provides its own light source in form of radio waves.Therefore it can “see” through darkness , clouds , rain , smoke etc.

Synthetic Aperture:key idea is that the satellite simulates a much larger antenna by combining radar returns collected as it moves, which gives it high spatial resolution despite using a physically small antenna.

SAR uses the motion of satellite, as satellite passes over a target, it sends rapid-fire sequence of radio pulses, and combining them cleverly tricks itself into thinking that it has a long antenna as a long antenna is required generally for capturing high resolution images from space making them hard to launch on a rocket practically.

main points- 24*7 vision(images look same all the time), weather proof (radio waves pass through so no effect) ,material sensing (very sensitive to surface roughness and moisture), change detection(very precise so photos taken weeks apart can be used to detect any minute changes).

Environmental Monitoring, Disaster Management, Maritime Surveillance, Infrastructure and Engineering.

Traditional satellites are like cameras but this SAR is like sonar with radio waves , it emits a pulse and measures the echo.

Polarization: (HH,VV,HV,VH): Describes the orientation of radio wave
VV/HH (Co-Polarized): good for detecting water (smooth surfaces)
HV/VH(Cross-polarized): good for detecting forest canopies(volume scattering).
VH/VV ratio is generally used as a feature in flood mapping and crop classification models.

Incidence angle:Angle at which the pulse hits a ground.when training a model on multi-scene SAR data, images from different incidence angles will have systematically different backscatter values for the same land cover.

For flood mapping: water surfaces appear very dark in SAR (specular reflection away from sensor), creating a strong contrast that's easy to threshold or train on.
