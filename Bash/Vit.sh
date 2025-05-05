#cd IFC
# N=1, obsevation =0.05s
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_1_Tprime_1.yaml"&&
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_1_Tprime_10.yaml"&&
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_1_Tprime_20.yaml"&&
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_1_Tprime_30.yaml"&&
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_1_Tprime_60.yaml"&&

# N=10, obsevation =0.05 * 10
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_10_Tprime_1.yaml"&&
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_10_Tprime_10.yaml"&&
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_10_Tprime_20.yaml"&&
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_10_Tprime_30.yaml"&&
python Train/train.py --config_yaml "Config/Vit_Full_Tasks/train_ifc_vit_N_10_Tprime_60.yaml"