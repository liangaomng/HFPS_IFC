
#cd IFC
# N=1, obsevation =0.05
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_1_Tprime_1.yaml"&&
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_1_Tprime_10.yaml"&&
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_1_Tprime_20.yaml"&&
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_1_Tprime_30.yaml"&&
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_1_Tprime_60.yaml"&&

# N=10, obsevation =0.05 * 10
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_10_Tprime_1.yaml"&&
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_10_Tprime_10.yaml"&&
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_10_Tprime_20.yaml"&&
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_10_Tprime_30.yaml"&&
python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_10_Tprime_60.yaml"

#sparse
# python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_10_Tprime_1_sparse_4.yaml"
# python Train/train.py --config_yaml "Config/Operator_Full_Tasks/train_ifc_opertor_N_10_Tprime_1_sparse_8.yaml"