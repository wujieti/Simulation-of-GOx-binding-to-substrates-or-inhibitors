#########################################################################
#                                                                       #
#                           Amber轨迹分析脚本                           #
#                            Edited by Ojet                             #
#                           Updated 20230718                            #
#                           wechat: ojet0501                            #
#                                                                       #
#########################################################################

import os
import subprocess
import glob
import shutil
import argparse
import configparser
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互模式
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
from matplotlib.colors import ListedColormap

#------------------------------设置运行参数--------------------------------------#
description = """
此脚本适用于AMBER分子动力学模拟轨迹分析，主要由CPPTRAJ和MMPBSA.py程序执行;

脚本默认:
(1)拓扑文件为cmp.prmtop;
(2)轨迹文件为cmp_eq1.mdcrd、cmp_eq2.mdcrd、...cmp_eq{i}.mdcrd、...、cmp_eq{time}.mdcrd;
(3)每段轨迹文件'cmp_eq{i}.mdcrd'的模拟时长为 1 ns ( 1000 帧 | 1000 ps );
(4)轨迹文件'cmp_eq1.mdcrd'至'cmp_eq{time}.mdcrd'总时间为 {time} ns;

使用示例：
nohup python MD-Analysis.py --complex 1-626 --receptor 1-625 --ligand 626 --time 0-100 --skip 100 --rmsd --mmgbsa --eq_time 40-100 --decomp &
(蛋白受体 1-625, 配体 626, 轨迹区间 0-100 ns, 从轨迹中每隔 100 帧(ps)采样, 计算RMSD、RMSF、B-facter, 计算 40-100 ns稳定段轨迹的'受体-配体'结合自由能和各残基能量贡献;)

nohup python MD-Analysis.py --complex 1-626 --receptor 1-625 --ligand 626 --mode 1 &
(蛋白受体 1-625, 配体 626, 采用分析“模式1”;)

nohup python MD-Analysis.py --complex 1-626 --receptor 1-625 --ligand 626 --all &
(蛋白受体 1-625, 配体 626, 运行此脚本包含的所有分析;)

nohup python MD-Analysis.py --solute 1-521  --mode 2 &
(单体蛋白 1-521, 采用分析“模式2”;)

nohup python MD-Analysis.py --solute 1-1000  --mode 3 &
(溶质为 1-1000 的多分子系统, 采用分析“模式3”;)

注意:--complex [ ] --receptor [ ] --ligand [ ] “受体-配体”系统 (必须设置);
     --solute [ ]  多分子系统 或 蛋白单体系统 (必须设置);

     --time 默认为 0-100 (ns) ;
     --skip 默认为 100 （需要设置为10的倍数）;
     --eq_time 默认为 60-100 (ns) , 用于在'平衡段轨迹'计算MMGBSA结合自由能;

            这几个参数可以在config.txt文档中设置, 如果准备了config.txt文档，运行时无需再设置，例：
            complex=1-626
            receptor=1-625
            ligand=626
            time=0-100
            skip=100
            eq_time=60-100

            运行：nohup python MD-Analysis.py --mode 1 &

     --mode [ ] 预设了3种分析模式，包含不同需求的分析设置:
       1 : 侧重“受体-配体”结合模型分析, 
         (提取Etot/Ek/Ep/Evdw/Eele数据, 计算RMSD/RMSF/B-facter/SASA/vdw+ele相互作用能/氢键/“受体-配体”接触矩阵/MMGBSA/残基能量, 提取模拟后平均结构);
       2 : 侧重蛋白构象变化分析, 
         (提取Etot/Ek/Ep/Evdw/Eele数据, 计算Rg/RMSD/RMSF/B-facter/SASA/氢键/二级结构/残基接触矩阵/动态相关性矩阵/主成分分析/帧间RMSD);
       3 ：侧重“多分子系统”在溶剂环境中的聚集行为分析, 
         (提取Etot/Ek/Ep/Evdw/Eele数据, 计算Rg/Density/RDF/MSD及扩散系数/氢键);                         

     --all 将执行所有分析, 时间成本比较高, 并且只适用于“受体-配体”结合体系;

     --restart 从"analysis-input"目录读取输入文件"*.in", 重新运行CPPTRAJ或MMPBSA.py程序;
               适用于修改输入文件"*.in"后重新分析轨迹;
               可以在"analysis-input"目录添加自定义(新)的输入文件"*.in",例如编写ptraj.in文件加入目录中;      

     --rdf 补充参数:
     --rdf_atm1 [ ] 设置需要计算RDF的原子或原子团(通常设置为某分子的中心原子,计算该分子的RDF),默认为 WAT@O;
     --rdf_atm2 [ ] 计算RDF时的参照原子或原子团(RDF原点),默认为 {complex}/{soulte}&!(@H=) 溶质表面; 

     --msd 补充参数:
     --msd_mol [ ] 设置需要计算MSD及扩散系数的原子团或分子或离子,可设置多个,空格隔断,默认为 Na+;

     运行将生成三个新目录：
     ./analysis-input/   保存各类分析的输入/控制文件;
     ./analysis-output/  保存各类分析的输出结果文件;
     ./analysis-log/     保存各类分析过程中的程序运行记录文件;
"""

# 解析命令行参数
parser = argparse.ArgumentParser(description='Analyze the simulation trajectory.', epilog=description, formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('--solute',  help='Mask for solute structure (e.g. 1-626);')
parser.add_argument('--complex',  help='Mask for complex structure (e.g. 1-626);')
parser.add_argument('--receptor', help='Mask for receptor structure (e.g. 1-625);')
parser.add_argument('--ligand', help='Mask for ligand structure (e.g. 626);')
parser.add_argument('--time', default='0-100', help='The total duration of the simulation trajectory, default is 0-100 (ns);')
parser.add_argument('--skip', type=int, default=10, help='The sampling interval of the trajectory, default is one frame per 10 (ps) interval;')
parser.add_argument('--eq_time', help='The equilibrium trajectory range used to calculate the binding free energy (MMGBSA method), defaults is 50-100 (ns);')
parser.add_argument('--avg', action='store_true', help='Extract the average structure;')
parser.add_argument('--energy', action='store_true', help='Extract the energy data of the simulation system (Etot,Ek,Ep,Evdw,Eele);')
parser.add_argument('--rmsd', action='store_true', help='Calculate RMSD,RMSDF,B-factor data of the simulation system;')
parser.add_argument('--density', action='store_true', help='Calculate Density of the simulation system;')
parser.add_argument('--sasa', action='store_true', help='Calculate SASA of the simulation system;')
parser.add_argument('--rg', action='store_true', help='Calculate Rg of the simulation system;')
parser.add_argument('--hbond', action='store_true', help='Hydrogen bonding analysis for the simulation system;')
parser.add_argument('--dssp', action='store_true', help='Secondary structure analysis for the simulation system;')
parser.add_argument('--nbi', action='store_true', help='Calculate the non-bonded interactions (ele and vdw) between the ligand and the surrounding residues (<4.5 Å);')
parser.add_argument('--pca', action='store_true', help='Principle component analysis for the simulation system;')
parser.add_argument('--rmsd_2d', action='store_true', help='Calculate 2D-RMSD matrix;')
parser.add_argument('--rdf', action='store_true', help='Calculate RDF of the simulation system;')
parser.add_argument('--rdf_atm1', default='WAT@O', help='Atoms required to calculate RDF, default is WAT@O;')
parser.add_argument('--rdf_atm2', default='{rdf_atm2_default}&!(@H=)', help='Reference atoms for RDF calculation, default is {complex} or {soulte}&!(@H=);')
parser.add_argument('--msd', action='store_true', help='Calculate MSD and Diffusion Contant of the simulation system;')
parser.add_argument('--msd_mol', action='append', nargs='+', default=[['Na+']], help='Specify molecules for MSD calculation, default is Na+;')
parser.add_argument('--contact_matrix', action='store_true', help='Calculate the residue contact matrix;')
parser.add_argument('--correlation_matrix', action='store_true', help='Calculate the dynamic cross-correlation matrix;')
parser.add_argument('--mmgbsa', action='store_true', help='Calculate the binding free energy between receptor and ligand;')
parser.add_argument('--decomp', action='store_true', help='Residue energy decomposition;')
parser.add_argument('--all', action='store_true', help='Run all calculations and analyses;')
parser.add_argument('--mode', choices=['1', '2', '3'], help='Mode option')
parser.add_argument("--restart", action="store_true", help="Restart Analysis")

args = parser.parse_args()

# 从 config.txt 中读取参数
config = {}
try:
    with open('config.txt') as file:
        for line in file:
            line = line.strip()
            if line and '=' in line:
                key, value = line.split('=', 1)
                config[key] = value
except FileNotFoundError:
    pass  # 如果 config.txt 文件不存在，跳过

args.solute = config.get('solute', args.solute)
args.complex = config.get('complex', args.complex)
args.receptor = config.get('receptor', args.receptor)
args.ligand = config.get('ligand', args.ligand)
args.time = config.get('time', args.time)
args.skip = int(config.get('skip', args.skip))
args.eq_time = config.get('eq_time', args.eq_time)


#--------------------------------检查运行参数------------------------------------#
time_start, time_end = args.time.split('-')
time_start = int(time_start)
time_end = int(time_end)
if args.eq_time:
    eq_start, eq_end = args.eq_time.split('-')
    eq_start = int(eq_start)
    eq_end = int(eq_end)
else:
    eq_start = int(time_end * 0.5)
    eq_end = int(time_end)    

skip = args.skip

# 检查受体配体或溶质序列设置是否合理,如果不合理终止程序;
if args.complex and args.receptor and args.ligand:
    if args.solute:
        print("错误:不能同时设置'--complex、--receptor、--ligand'和'--solute'。")
        print("使用示例：python MD-Analysis.py --complex 1-626 --receptor 1-625 --ligand 626")
        print("          python MD-Analysis.py --solute 1-626")
        sys.exit(1)  # 终止脚本并返回错误代码1
    else:
        complex = args.complex
        receptor = args.receptor
        ligand = args.ligand
        complex_or_solute = args.complex
        first_residue, last_residue = args.complex.split('-')
        first_residue = int(first_residue)
        last_residue = int(last_residue)
        print(f'复合结构为: {complex};')
        print(f'受体结构为: {receptor};')
        print(f'配体结构为: {ligand};')
        print(f'模拟轨迹总时长为: {time_end} ns;')
        print(f'采样间隔为: {skip} ps;')
        # 更新 args.rdf_atm2 的默认值
        args.rdf_atm2 = args.rdf_atm2.format(args=args, rdf_atm2_default=args.complex)
elif args.solute:
    if args.complex and args.receptor and args.ligand:
        print("错误:不能同时设置'--complex、--receptor、--ligand'和'--solute'。")
        print("使用示例：python MD-Analysis.py --complex 1-626 --receptor 1-625 --ligand 626")
        print("          python MD-Analysis.py --solute 1-626")
        sys.exit(1)
    else:
        solute = args.solute
        complex_or_solute = args.solute 
        first_residue, last_residue = args.solute.split('-')
        first_residue = int(first_residue)
        last_residue = int(last_residue)
        print(f'溶质结构为: {solute};')
        print(f'模拟轨迹总时长为: {time_end} ns;')
        print(f'采样间隔为: {skip} ps;')
        # 更新 args.rdf_atm2 的默认值
        args.rdf_atm2 = args.rdf_atm2.format(args=args, rdf_atm2_default=args.solute)
else:
    print("错误:必须设置'--complex、--receptor、--ligand'或'--solute'两组参数其中之一。")
    print("使用示例：python MD-Analysis.py --complex 1-626 --receptor 1-625 --ligand 626")
    print("          python MD-Analysis.py --solute 1-626")
    sys.exit(1)

# 检查模拟轨迹采样范围及间隔设置是否合理,如果不合理终止程序;
if skip > 1000:
    print("错误:采样间隔过大,请重新设置'--skip'参数。")
    sys.exit(1)

if eq_end > time_end or eq_start < time_start:
    print(f"错误:平衡段轨迹超出了总模拟轨迹时间范围 {time_start}-{time_end} ns,请重新设置'--eq_time'参数。")
    sys.exit(1)

# 检查平衡段轨迹区间;
if args.mmgbsa:
    print(f"平衡段轨迹区间为：{eq_start}-{eq_end} ns;")

# 检查是否有缺失文件,如果有缺失文件终止程序;
required_files = ["cmp.prmtop", "cmp.inpcrd"]
for i in range(1, time_end + 1):
    required_files.append(f"cmp_eq{i}.mdcrd")

missing_files = [file for file in required_files if not os.path.exists(file)]
if missing_files:
    print(f"缺少以下文件:{missing_files}")
    sys.exit(1)


#---------------------------设置输入、输出文件路径------------------------------#
# 创建一个新目录用于保存cpptraj程序所有输入（控制）文件;
Analysis_Input_Folder = "analysis-input"
os.makedirs(Analysis_Input_Folder, exist_ok=True)

# 创建一个新目录用于保存cpptraj程序的运行日志log文件;
Analysis_Log_Folder = "analysis-log"
os.makedirs(Analysis_Log_Folder, exist_ok=True)

# 创建一个新目录用于保存cpptraj程序的输出文件;
Analysis_Output_Folder = "analysis-output"
os.makedirs(Analysis_Output_Folder, exist_ok=True)


#---------------------------------其他设置-----------------------------------#
# 纳秒转换为帧数（1 帧 = 1 ps);
frame_start = int(time_start * 1000) 
frame_end  = int(time_end * 1000)

# 设置绘图时间单位;
unit_for_time = "{:.0e}".format(skip / 1000) # 设置一下时间轴单位
print(f"所有输出数据的时间(Time)单位为: ×{unit_for_time} ns;")
exponent = int(unit_for_time.split('e')[1])
unit_for_time = fr'$10^{{{exponent}}}$'

# 设置cpptraj轨迹读取集;
trajectory_files=[f"trajin cmp_eq{i}.mdcrd 1 last {skip}\n" for i in range(time_start + 1, time_end + 1)] 

# 设置'平衡段'轨迹读取集;
equilibrium_segment_trajectory_files=[f"trajin cmp_eq{i}.mdcrd 1 last {skip}\n" for i in range(eq_start + 1, eq_end + 1)]

# 设置强制刷新缓存;
print()
sys.stdout.flush()


#---------------------------------设置模块-----------------------------------#
# CPPTRAJ程序运行模块;
def run_cpptraj(input_filename):
    input_file = os.path.join(Analysis_Input_Folder, input_filename)
    log_filename = input_filename.replace(".in", ".log")
    log_file = os.path.join(Analysis_Log_Folder, log_filename)
    cpptraj_command = f'cpptraj -i {input_file} > {log_file}'
    subprocess.run(cpptraj_command, shell=True)

# ante-MMPBSA.py程序运行模块;
def run_ante_mmpbsa():
    commands_prepare_for_mmpbsa = [
        "rm -f com.prmtop",
        "rm -f rec.prmtop",
        "rm -f lig.prmtop",
        f"ante-MMPBSA.py -p cmp.prmtop -c com.prmtop -s \":WAT,Na+,Cl-\" --radii=mbondi3",
        f"ante-MMPBSA.py -p cmp.prmtop -c rec.prmtop -s \":WAT,Na+,Cl-,{ligand},\" --radii=mbondi3",
        f"ante-MMPBSA.py -p cmp.prmtop -c lig.prmtop -s \":WAT,Na+,Cl-,{receptor},\" --radii=mbondi3"
    ]
    for command in commands_prepare_for_mmpbsa:
        subprocess.run(command, shell=True)

# MMPBSA.py程序运行模块;
def run_mmpbsa(input_filename):
    input_file = os.path.join(Analysis_Input_Folder, input_filename)
    log_filename = input_filename.replace(".in", ".log")
    log_file = os.path.join(Analysis_Log_Folder, log_filename)
    mmpbsa_command = f'$AMBERHOME/bin/MMPBSA.py -O -i {input_file} -o MMGBSA_BindingEnergy.txt -eo MMGBSA_BindingEnergy_EveryFrame.csv -do MMGBSA_EnergyDecomp.txt -deo MMGBSA_EnergyDecomp_EveryFrame.csv -sp cmp.prmtop -cp com.prmtop -rp rec.prmtop -lp lig.prmtop -y cmp_eq{eq_start}-{eq_end}_image.mdcrd > {log_file}'
    subprocess.run(mmpbsa_command, shell=True)

# 整理输出数据模块 (插入分隔符和列标题、重新保存为txt或csv文件);
def insert_delimiter_and_headers(input_file, output_file, start_line, column_headers=None):
    txt_content = []
    csv_content = []

    with open(input_file, 'r') as file:
        lines = file.readlines()

        for line in lines[start_line:]:
            extract_data = line.strip().split()

            txt_line = '\t'.join(extract_data)
            txt_content.append(txt_line)
            
            delimiter = ','
            csv_line = delimiter.join(extract_data)
            csv_content.append(csv_line)

    if output_file.endswith('.txt'):
        with open(output_file, 'w') as file:
            if column_headers is not None:
                file.write('\t'.join(column_headers) + '\n')
            file.write('\n'.join(txt_content))

    elif output_file.endswith('.csv'):
        with open(output_file, 'w') as file:
            if column_headers is not None:
                file.write(','.join(column_headers) + '\n')
            file.write('\n'.join(csv_content))

# 整理输出数据模块 (仅修改第一行第一列的内容,即索引名称);
def change_index_name(filename, new_content):
    with open(filename, 'r') as file:
        lines = file.readlines()   
    if lines:
        columns = lines[0].split() 
        if columns:
            columns[0] = new_content
            lines[0] = " ".join(columns) + '\n'  
    # 将修改后的内容写回文件
    with open(filename, 'w') as file:
        file.writelines(lines)

# 作图模块（曲线或直方图）;
def plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list=['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'orange', 'brown', 'gray'], top_margin=0.3, bottom_margin=0.3, plot_type='curve', ):
    # 读取CSV文件; 
    columns = []
    for csv_tuple in csv_files:
        csv_file, col_name = csv_tuple
        df = pd.read_csv(csv_file)
        columns.append(df[col_name])  
    data = pd.concat(columns, axis=1)  
    print(data)
    print()
    # 解析数据; 
    x_values = data.iloc[:,0]
    xmin = x_values.min()
    xmax = x_values.max()
    y_values = data.iloc[:,1:]
    ymin = y_values.min().min()
    ymax = y_values.max().max()
    # 检查设置;
    if len(data.columns[1:]) != len(curve_label):
        raise ValueError("The number of curve labels and the Y-coordinate columns do not match. Please check and reset!")
    if len(color_list) < len(data.columns[1:]):
        raise ValueError("The number of color list and the Y-coordinate columns do not match. Please check and reset!")
    # 创建曲线或柱状图;
    plt.figure(figsize=(30, 20))    
    for column, label, color in zip(np.arange(1,len(data.columns[1:]) + 1), curve_label, color_list):
        if plot_type == "curve":
            plt.plot(data.iloc[:,0], data.iloc[:,column], label=label, linewidth=4, color=color,)
        elif plot_type == "bar":
            plt.bar(data.iloc[:,0], data.iloc[:,column], label=label, linewidth=4, color=color,)
        else:
            raise ValueError("Please set the correct plot type (plot_type='curve' or 'bar')!")
    # 设置图例;
    if len(data.columns[1:]) > 1:    
        plt.legend(fontsize=48, frameon=False, loc='upper left', bbox_to_anchor=(1.02, 1), ) 
    # 设置标题;
    plt.title(title, fontsize=56, pad=50)
    # 设置xy轴标签;
    plt.xlabel(x_label, fontsize=50, labelpad=40)
    plt.ylabel(y_label, fontsize=50, labelpad=30)
    # 设置xy轴刻度标签;
    plt.xticks(fontsize=48)
    plt.yticks(fontsize=48)
    # 设置xy轴显示范围;
    if plot_type == "curve":
        if 'Time' in data.columns[0]:
            plt.xlim(0, xmax)
        else:
            plt.xlim(xmin, xmax)
        if abs(ymin) < 10:
            plt.ylim(0, ymax + top_margin * (ymax - ymin)) 
        else:
            plt.ylim(ymin - bottom_margin * (ymax - ymin), ymax + top_margin * (ymax - ymin))
    if plot_type == "bar":
        if ymin < 0:
            plt.axhline(y=0, color='black', linestyle='-', linewidth=4, ) # 加水平轴线;
            plt.ylim(ymin - bottom_margin * (ymax - ymin), ymax + top_margin * (ymax - ymin))
        else:
            plt.ylim(0, ymax + top_margin * (ymax - ymin))
    # 设置xy轴边框线条粗细;
    ax = plt.gca()  
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4) 
    ax.spines['bottom'].set_linewidth(4) 
    ax.spines['left'].set_linewidth(4) 
    # 设置xy轴主、次刻度线粗细;
    plt.tick_params(axis='both', which='major', width=4, length=10)
    plt.tick_params(axis='both', which='minor', width=2, length=5) 
    # 保存图片为TIF文件;
    plt.savefig(tif_file, dpi=300, bbox_inches='tight', format='tif')

# 作图模块（平均值作柱状图）;
def plot_avg_bar(csv_files, tif_file, title, x_label, y_label, x_ticks_labels, color_list=['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'orange', 'brown', 'gray'], top_margin=0.3, bottom_margin=0.3, int_yvalue=False ):
    # 读取CSV文件;
    columns = []  
    for csv_tuple in csv_files:
        csv_file, col_name = csv_tuple 
        df = pd.read_csv(csv_file)
        mean_value = df[col_name].mean()
        columns.append((col_name, mean_value)) 
    data = pd.DataFrame(columns, columns=['Set', 'Avg'])
    print(data)
    print()
    # 解析数据;
    ymin = data['Avg'].min()
    ymax = data['Avg'].max()
    # 检查设置;
    if len(data['Avg']) != len(x_ticks_labels):
        raise ValueError("The number of X ticks labels and data do not match. Please check and reset!")
    # 创建直方图;
    plt.figure(figsize=(30, 20))
    plt.bar(np.arange(len(data.index)), data['Avg'], color=color_list,)
    # 设置xy轴显示范围;
    if ymin < 0:
        plt.axhline(y=0, color='black', linestyle='-', linewidth=4, ) # 加水平轴线;
        plt.ylim(ymin - bottom_margin * (ymax - ymin), ymax + top_margin * (ymax - ymin))
    else:
        plt.ylim(0, ymax + top_margin * (ymax - ymin))
    # 标注Y值;
    for index, row in data.iterrows():
        va = 'top' if row['Avg'] < 0 else 'bottom'
        if int_yvalue:
            y_value = int(row['Avg'])  # 转换为整数
            plt.text(index, row['Avg'], f"{y_value}", ha='center', va=va, fontsize=48)
        else:
            plt.text(index, row['Avg'], f"{row['Avg']:.2f}", ha='center', va=va, fontsize=48)
    # 设置标题;
    plt.title(title, fontsize=56, pad=50)
    # 设置xy轴标签;
    plt.ylabel(y_label, fontsize=50, labelpad=40)
    # 设置xy轴刻度标签;
    plt.xticks(np.arange(len(data.index)), x_ticks_labels, fontsize=48,)
    plt.tick_params(axis='x', pad=30)
    plt.yticks(fontsize=48)
    # 设置xy轴边框线条粗细;
    ax = plt.gca()  
    ax.spines['top'].set_linewidth(4)
    ax.spines['right'].set_linewidth(4) 
    ax.spines['bottom'].set_linewidth(4) 
    ax.spines['left'].set_linewidth(4) 
    # 设置xy轴主、次刻度线粗细;
    plt.tick_params(axis='both', which='major', width=4, length=10)
    plt.tick_params(axis='both', which='minor', width=2, length=5) 
    # 保存图片为TIF文件;
    plt.savefig(tif_file, dpi=300, bbox_inches='tight', format='tif')

# 作图模块（使用矩阵式数据作热图）;
def plot_matrix_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label, colorbar_cmap=None, colorbar_custom_cmap=['white', 'red', 'yellow', 'blue', 'green', 'purple', 'pink', 'orange', 'brown', 'gray'], index_col=0, header=0, transpose_matrix=True,):
    # 从CSV文件读取数据;
    data_tmp = pd.read_csv(csv_file, index_col=index_col, header=header,)
    # 解析数据;
    data = data_tmp.T if transpose_matrix else data_tmp
    z_values = data.values.astype(float)
    zmin = z_values.min().min()
    zmax = z_values.max().max()
    print(data)
    print()
    # 创建热图;
    fig, ax = plt.subplots(figsize=(30, 20))
    if colorbar_cmap is not None:
        if zmin < 0:
            colorbar_lower_limit = -zmax
        else:
            colorbar_lower_limit = 0
        heatmap = ax.imshow(data, cmap=colorbar_cmap, aspect='auto', vmin=colorbar_lower_limit, vmax=zmax,)
        colorbar = plt.colorbar(heatmap)
    else:
        # 自定义色板创建热图;
        my_custom_cmap = ListedColormap(colorbar_custom_cmap)
        number_of_custom_colors = len(colorbar_custom_cmap)
        select_colors = int(zmax + 1)
        my_custom_cmap_subset = ListedColormap(my_custom_cmap.colors[:select_colors])
        heatmap = ax.imshow(data, cmap=my_custom_cmap_subset, aspect='auto', vmin=-0.5, vmax=zmax+0.5,)
        colorbar = plt.colorbar(heatmap, ticks=np.arange(select_colors)) 
    # 设置colorbar标签;
    colorbar.set_label(colorbar_label, fontsize=48, labelpad=60, rotation=-90)
    # 设置colorbar刻度标签;
    colorbar.ax.yaxis.set_tick_params(labelsize=48)
    if 'Secondary Structure Type' in colorbar_label:
        colorbar.ax.set_yticklabels(['None', 'Para', 'Anti', '3-10', 'Alpha', 'Pi', 'Turn', 'Bend'])
    # 设置colorbar刻度线的长度和粗细;
    colorbar.ax.tick_params(axis='y', which='both', direction='in', length=10, width=4) 
    colorbar.ax.tick_params(axis='x', which='both', direction='in', length=0) 
    # 设置colorbar边框线条粗细;
    colorbar.outline.set_linewidth(4)
    # 设置轴标签;
    ax.set_xlabel(x_label, fontsize=50, labelpad=40)
    ax.set_ylabel(y_label, fontsize=50, labelpad=30)
    # 设置标题;
    ax.set_title(title, fontsize=56, pad=50)
    # 设置x轴刻度与标签;
    if len(data.columns) < 29:
        ax.set_xticks(np.arange(len(data.columns)))    
        ax.set_xticklabels(data.columns, fontsize=48) 
    else:
        xticklabels_gap = int(len(data.columns) / 10)
        if 'Time' in x_label:
            ax.set_xticks(np.arange(-0.5, len(data.columns), xticklabels_gap))
            renumber_xticklabels = np.arange(0, len(data.columns)+1, xticklabels_gap)
            ax.set_xticklabels(renumber_xticklabels, fontsize=48)
        else:
            ax.set_xticks(np.arange(0, len(data.columns)+1, xticklabels_gap))
            xticklabels_indices = np.arange(0, len(data.columns), xticklabels_gap)
            renumber_xticklabels = [str(data.columns[i]) for i in xticklabels_indices]
            ax.set_xticks(xticklabels_indices)
            ax.set_xticklabels(renumber_xticklabels, fontsize=48)
    # 设置y轴刻度与标签;
    if len(data.index) < 29:
        ax.set_yticks(np.arange(len(data.index)))    
        ax.set_yticklabels(data.index, fontsize=48)    
    else:
        yticklabels_gap = int(len(data.index) / 10)
        if 'Time' in y_label:
            ax.set_yticks(np.arange(-0.5, len(data.index), yticklabels_gap))
            renumber_yticklabels = np.arange(0, len(data.index)+1, yticklabels_gap)
            ax.set_yticklabels(renumber_yticklabels, fontsize=48)
        else:   
            ax.set_yticks(np.arange(0, len(data.index)+1, yticklabels_gap))
            yticklabels_indices = np.arange(0, len(data.index), yticklabels_gap)
            renumber_yticklabels = [str(data.index[i]) for i in yticklabels_indices]
            ax.set_yticks(yticklabels_indices)
            ax.set_yticklabels(renumber_yticklabels, fontsize=48)
    # 设置xy轴显示范围;
    ax.set_xlim(-0.5 , len(data.columns)-0.5 )
    ax.set_ylim(-0.5 , len(data.index)-0.5 )
    # 设置xy轴边框线条粗细;
    ax.spines['top'].set_linewidth(4)  
    ax.spines['right'].set_linewidth(4)  
    ax.spines['bottom'].set_linewidth(4) 
    ax.spines['left'].set_linewidth(4) 
    # 设置xy轴主、次刻度线粗细;
    ax.tick_params(axis='x', which='major', direction='in', length=10, width=4,)
    ax.tick_params(axis='x', which='minor', direction='in', length=5, width=2,)  
    ax.tick_params(axis='y', which='major', direction='in', length=10, width=4,) 
    ax.tick_params(axis='y', which='minor', direction='in', length=5, width=2,)
    # 设置x轴刻度和标签显示在底部;
    ax.xaxis.set_ticks_position('bottom')
    # 保存热图为TIF文件;
    plt.savefig(tif_file, dpi=600, bbox_inches='tight', format='tif')

# 作图模块（使用xyz三列式数据作热图）;
def plot_xyz_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label, colorbar_cmap=None, x=0, y=1, z=2,):
    # 从CSV文件读取数据;
    data = pd.read_csv(csv_file, header=None, skiprows=[0],)
    x_values = data.iloc[:, x].values
    y_values = data.iloc[:, y].values
    z_values = data.iloc[:, z].values
    zmin = z_values.min()
    zmax = z_values.max()
    # 获取x和y的唯一值以创建网格;
    unique_x = np.unique(x_values).astype(int)
    unique_y = np.unique(y_values).astype(int)
    # 创建一个网格矩阵来存储z值;
    grid_z = np.zeros((len(unique_y), len(unique_x)))
    # 将z值填充到网格矩阵中
    for i in range(len(z_values)):
        x_index = np.where(unique_x == x_values[i])[0][0]
        y_index = np.where(unique_y == y_values[i])[0][0]
        grid_z[y_index, x_index] = z_values[i]
    print(f"x_values = {unique_x}")
    print(f"y_values = {unique_y}")
    print(f"z_values = {grid_z}")
    print()
    # 绘制热图
    fig, ax = plt.subplots(figsize=(30, 20))
    if colorbar_cmap is not None:
        if zmin < 0:
            colorbar_lower_limit = -zmax
        else:
            colorbar_lower_limit = 0
        heatmap = ax.imshow(grid_z, cmap=colorbar_cmap, aspect='auto', vmin=0, vmax=zmax,)
        colorbar = plt.colorbar(heatmap)
    # 设置colorbar标签;
    colorbar.set_label(colorbar_label, fontsize=48, labelpad=60, rotation=-90)
    # 设置colorbar刻度标签;
    colorbar.ax.yaxis.set_tick_params(labelsize=48)
    # 设置colorbar刻度线的长度和粗细;
    colorbar.ax.tick_params(axis='y', which='both', direction='in', length=10, width=4) 
    colorbar.ax.tick_params(axis='x', which='both', direction='in', length=0) 
    # 设置colorbar边框线条粗细;
    colorbar.outline.set_linewidth(4)
    # 设置轴标签;
    ax.set_xlabel(x_label, fontsize=50, labelpad=40)
    ax.set_ylabel(y_label, fontsize=50, labelpad=30)
    # 设置标题;
    ax.set_title(title, fontsize=56, pad=50)
    # 设置x轴刻度与标签;
    if len(unique_x) < 29:
        ax.set_xticks(np.arange(len(unique_x)))    
        ax.set_xticklabels(unique_x, fontsize=48) 
    else:
        xticklabels_gap = int(len(unique_x) / 10)
        if 'Time' in x_label:
            ax.set_xticks(np.arange(-0.5, len(unique_x), xticklabels_gap))
            renumber_xticklabels = np.arange(0, len(unique_x)+1, xticklabels_gap)
            ax.set_xticklabels(renumber_xticklabels, fontsize=48)
        else:
            ax.set_xticks(np.arange(0, len(unique_x)+1, xticklabels_gap))
            xticklabels_indices = np.arange(0, len(unique_x), xticklabels_gap)
            renumber_xticklabels = [str(unique_x[i]) for i in xticklabels_indices]
            ax.set_xticks(xticklabels_indices)
            ax.set_xticklabels(renumber_xticklabels, fontsize=48)
    # 设置y轴刻度与标签;
    if len(unique_y) < 29:
        ax.set_yticks(np.arange(len(unique_y)))    
        ax.set_yticklabels(unique_y, fontsize=48)    
    else:
        yticklabels_gap = int(len(unique_y) / 10)
        if 'Time' in y_label:
            ax.set_yticks(np.arange(-0.5, len(unique_y), yticklabels_gap))
            renumber_yticklabels = np.arange(0, len(unique_y)+1, yticklabels_gap)
            ax.set_yticklabels(renumber_yticklabels, fontsize=48)
        else:   
            ax.set_yticks(np.arange(0, len(unique_y)+1, yticklabels_gap))
            yticklabels_indices = np.arange(0, len(unique_y), yticklabels_gap)
            renumber_yticklabels = [str(unique_y[i]) for i in yticklabels_indices]
            ax.set_yticks(yticklabels_indices)
            ax.set_yticklabels(renumber_yticklabels, fontsize=48)
    # 设置xy轴显示范围;
    ax.set_xlim(-0.5 , len(unique_x)-0.5 )
    ax.set_ylim(-0.5 , len(unique_y)-0.5 )
    # 设置xy轴边框线条粗细;
    ax.spines['top'].set_linewidth(4)  
    ax.spines['right'].set_linewidth(4)  
    ax.spines['bottom'].set_linewidth(4) 
    ax.spines['left'].set_linewidth(4) 
    # 设置xy轴主、次刻度线粗细;
    ax.tick_params(axis='x', which='major', direction='in', length=10, width=4,)
    ax.tick_params(axis='x', which='minor', direction='in', length=5, width=2,)  
    ax.tick_params(axis='y', which='major', direction='in', length=10, width=4,) 
    ax.tick_params(axis='y', which='minor', direction='in', length=5, width=2,)
    # 设置x轴刻度和标签显示在底部;
    ax.xaxis.set_ticks_position('bottom')
    # 保存热图为TIF文件;
    plt.savefig(tif_file, dpi=600, bbox_inches='tight', format='tif')


#----------------------------------平均结构提取---------------------------------#    
if args.avg or args.all or args.mode == '1' or args.mode == '2':

    # 编写cpptraj程序输入/控制文件;
    Input_File_Path = os.path.join(Analysis_Input_Folder, "average_structure.in")
    with open(Input_File_Path, "w") as f:
        content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 输出平衡段轨迹的平均结构;
average Avg.pdb pdb
"""
        f.write(content)
    
    # 运行cpptraj程序
    run_cpptraj('average_structure.in')
    
    # 编写cpptraj程序输入/控制文件;
    Input_File_Path = os.path.join(Analysis_Input_Folder, "rmsd_average_structure.in")
    with open(Input_File_Path, "w") as f:
        content = f"""
# 载入拓扑文件prmtop轨迹文件mdcrd;
parm cmp.prmtop
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 以平均结构为为参照，计算RMSD值;
reference Avg.pdb
rms reference out rmsd_to_Avg_structure.txt :{complex_or_solute}&!(@H=)
"""
        f.write(content)
    
    # 运行cpptraj程序
    run_cpptraj('rmsd_average_structure.in')
    
    # 读取 rmsd_to_Avg_structure.txt 文件内容;
    with open("rmsd_to_Avg_structure.txt", "r") as file:
        lines = file.readlines()
    
    # 根据第二列数据(以平均结构为为参照计算的RMSD值), 进行排序（不含标题行）;
    sorted_lines = sorted(lines[1:], key=lambda line: float(line.split()[1]))
    
    # 输出第二列数据最小值对应帧数;
    if sorted_lines:
        fields = sorted_lines[0].split()
        Frame_for_outPDB = fields[0]

    # 编写cpptraj程序输入/控制文件;
    Input_File_Path = os.path.join(Analysis_Input_Folder, "representative_structure.in")
    with open(Input_File_Path, "w") as f:
        content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 删除溶剂和离子
strip :WAT,Na+,Cl-
# 输出平衡段轨迹的平均结构;
trajout Avg_structure.pdb onlyframes {Frame_for_outPDB} 
"""
        f.write(content)
    
    # 运行cpptraj程序
    run_cpptraj('representative_structure.in')
    os.remove("Avg.pdb")
    print(f"平均结构提取完成(第{Frame_for_outPDB}帧);")
    sys.stdout.flush()

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") + glob.glob("Avg_structure.pdb") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#--------------------------提取Etot\Ek\Ep\Evdw\Eele数据-------------------------#
if args.energy or args.all or args.mode == '1' or args.mode == '2' or args.mode == '3':
    # 创建数据集列表;
    Etot_data = []
    Ek_data = []
    Ep_data = []
    Evdw_data = []
    Eele_data = []

    # 从out文件提取各类数据，保存到数据集列表中;
    for i in range(time_start + 1, time_end + 1):
        input_file_name = f"cmp_eq{i}.out"  # 指定输入文件名;
        with open(input_file_name, "r") as input_file:
            lines = input_file.readlines()
        for line in lines:
            if "Etot" in line:
                values = line.split()
                Etot_data.append(values[2])
                Ek_data.append(values[5])
                Ep_data.append(values[8])
            if "VDWAALS" in line:
                values = line.split()
                Evdw_data.append(values[10])
            if "EELEC" in line:
                values = line.split()
                Eele_data.append(values[2])
            elif "A V" in line:
                break

    print("Etot,Ek,Ep,Evdw,Eele数据提取完成;")
    sys.stdout.flush()

    # 保存数据到CSV文件
    with open('Ep_Ek_Evdw_Eele_Etot.csv', 'w') as csv_file:
        csv_file.write("Time,Ep (kcal/mol),Ek (kcal/mol),Evdw (kcal/mol),Eele (kcal/mol),Etot (kcal/mol)\n")
        for a, b in zip(range(1, int((frame_end - frame_start) / skip) + 1), range(0, len(Etot_data), skip)):
            csv_file.write(f'{a},{Ep_data[b]},{Ek_data[b]},{Evdw_data[b]},{Eele_data[b]},{Etot_data[b]}\n')
    
    # 绘图(Ep_Ek_Evdw_Eele_Etot.tif);
    csv_files = [
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Time"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Ep (kcal/mol)"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Ek (kcal/mol)"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Evdw (kcal/mol)"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Eele (kcal/mol)"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Etot (kcal/mol)"),
    ]
    tif_file = 'Ep_Ek_Evdw_Eele_Etot.tif'
    title = 'System Potential/Kinetic/Van der Waals/Electrostatic/Total Energy'
    x_label = f'Time (×{unit_for_time} ns)'
    y_label = 'Energy (kcal/mol)'
    curve_label = ['Ep','Ek', 'Evdw', 'Eele', 'Etot',]
    color_list=['#16499D', '#E71F19', '#36AE37', '#7D4195', '#EF7D1A']
    plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list,top_margin=0.3, bottom_margin=0.3,)


    # 绘图(Avg_Ep_Ek_Evdw_Eele_Etot.tif);
    csv_files = [
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Ep (kcal/mol)"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Ek (kcal/mol)"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Evdw (kcal/mol)"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Eele (kcal/mol)"),
        ("Ep_Ek_Evdw_Eele_Etot.csv", "Etot (kcal/mol)"),
    ]
    tif_file = "Avg_Ep_Ek_Evdw_Eele_Etot.tif"
    title = "Average Value of System Energies"
    x_label = "Energy Term" 
    y_label = "Avg. Energy (kcal/mol) "
    x_ticks_labels = ['Ep', 'Ek', 'Evdw', 'Eele', 'Etot',]
    color_list=['#16499D', '#E71F19', '#36AE37', '#7D4195', '#EF7D1A']
    plot_avg_bar(csv_files, tif_file, title, x_label, y_label, x_ticks_labels, color_list)

    # 将结果转移至analysis-output目录;
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#--------------------------计算RMSD\RMSDF\B-factor数据--------------------------#
if args.rmsd or args.all or args.mode == '1' or args.mode == '2' or args.mode == '3':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "rmsd.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算Complex的RMSD、RMSF、B-factor;
rms mass first out RMSD_Complex.txt :{complex}&!(@H=)
atomicfluct out RMSF_Complex.txt :{complex}&!(@H=) byres
atomicfluct out Bfactor_Complex.txt :{complex}&!(@H=) byres bfactor
# 计算Receptor和Ligand的RMSD;
rms mass first out RMSD_Ligand.txt :{ligand}&!(@H=)
# 计算Pocket残基（距离Ligand 4.5埃）的RMSD、RMSF、B-factor;
reference cmp.inpcrd
rms mass first out RMSD_Pocket.txt ":{ligand}<:4.5 & !:WAT & !:Na+ & !:Cl- & !:{ligand}"
atomicfluct out RMSF_Pocket.txt ":{ligand}<:4.5 & !:WAT & !:Na+ & !:Cl- & !:{ligand}" byres
atomicfluct out Bfactor_Pocket.txt ":{ligand}<:4.5 & !:WAT & !:Na+ & !:Cl- & !:{ligand}" byres bfactor
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "rmsd.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算Solute的RMSD、RMSF、B-factor;
rms mass first out RMSD.txt :{solute}&!(@H=)
atomicfluct out RMSF.txt :{solute}&!(@H=) byres
atomicfluct out Bfactor.txt :{solute}&!(@H=) byres bfactor
"""
            f.write(content)

    # 运行cpptraj程序;
    run_cpptraj('rmsd.in')
    print("RMSD,RMSDF,B-factor数据计算完成;")
    sys.stdout.flush()

    if args.receptor and args.ligand:
        # 整理cpptraj输出数据;
        insert_delimiter_and_headers("RMSD_Complex.txt", "RMSD_Complex.csv", 1, ["Time", "RMSD_Complex (Å)"])
        insert_delimiter_and_headers("RMSD_Ligand.txt", "RMSD_Ligand.csv", 1, ["Time", "RMSD_Ligand (Å)"])
        insert_delimiter_and_headers("RMSD_Pocket.txt", "RMSD_Pocket.csv", 1, ["Time", "RMSD_Pocket (Å)"])
        insert_delimiter_and_headers("RMSF_Complex.txt", "RMSF_Complex.csv", 1, ["Residue No.", "RMSF_Complex (Å)"])
        insert_delimiter_and_headers("RMSF_Pocket.txt", "RMSF_Pocket.csv", 1, ["Residue No.", "RMSF_Pocket (Å)"])
        insert_delimiter_and_headers("Bfactor_Complex.txt", "Bfactor_Complex.csv", 1, ["Residue No.", "Bfactor_Complex (Å)"])
        insert_delimiter_and_headers("Bfactor_Pocket.txt", "Bfactor_Pocket.csv", 1, ["Residue No.", "Bfactor_Pocket (Å)"])

        # 绘图(RMSD.tif);
        csv_files = [
            ("RMSD_Complex.csv","Time"),
            ("RMSD_Complex.csv","RMSD_Complex (Å)"),
            ("RMSD_Ligand.csv","RMSD_Ligand (Å)"),
            ("RMSD_Pocket.csv","RMSD_Pocket (Å)"),
        ]
        tif_file = 'RMSD.tif'
        title = 'RMSD of Complex/Ligand/Pocket Heavy Atoms'
        x_label = f'Time (×{unit_for_time} ns)'
        y_label = 'RMSD (Å)'
        curve_label = ['Complex','Ligand','Pocket']
        color_list=['#16499D', '#E71F19', '#36AE37',] 
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3,)

        # 绘图(RMSF.tif);
        csv_files = [
            ("RMSF_Complex.csv","Residue No."),
            ("RMSF_Complex.csv","RMSF_Complex (Å)"),
        ]
        tif_file = 'RMSF.tif'
        title = 'RMSF of Complex Heavy Atoms'
        x_label = 'Residue No.'
        y_label = 'RMSF (Å)'
        curve_label = ['Complex']
        color_list = ['#16499D'] 
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.1, bottom_margin=0.3,)

        # 绘图(Bfactor.tif);
        csv_files = [
            ("Bfactor_Complex.csv","Residue No."),
            ("Bfactor_Complex.csv","Bfactor_Complex (Å)"),
        ]
        tif_file = 'Bfactor.tif'
        title = 'Bfactor of Complex Heavy Atoms'
        x_label = 'Residue No.'
        y_label = 'Bfactor (Å)'
        curve_label = ['Complex']
        color_list = ['#16499D'] 
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.1, bottom_margin=0.3,)      

    if args.solute:
        # 整理cpptraj输出数据;
        insert_delimiter_and_headers("RMSD.txt", "RMSD.csv", 1, ["Time", "RMSD (Å)"])
        insert_delimiter_and_headers("RMSF.txt", "RMSF.csv", 1, ["Residue No.", "RMSF (Å)"])
        insert_delimiter_and_headers("Bfactor.txt", "Bfactor.csv", 1, ["Residue No.", "Bfactor (Å)"])

        # 绘图(RMSD.tif);
        csv_files = [
            ("RMSD.csv","Time"),
            ("RMSD.csv","RMSD (Å)"),
        ]
        tif_file = 'RMSD.tif'
        title = 'RMSD of Solute Heavy Atoms'
        x_label = f'Time (×{unit_for_time} ns)'
        y_label = 'RMSD (Å)'
        curve_label = ['Solute']
        color_list = ['#16499D']
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3,)

        # 绘图(RMSF.tif);
        csv_files = [
            ("RMSF.csv","Residue No."),
            ("RMSF.csv","RMSF (Å)"),
        ]
        tif_file = 'RMSF.tif'
        title = 'RMSF of Solute Heavy Atoms'
        x_label = 'Residue No.'
        y_label = 'RMSF (Å)'
        curve_label = ['Solute']
        color_list = ['#16499D'] 
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3,)

        # 绘图(Bfactor.tif);
        csv_files = [
            ("Bfactor.csv","Residue No."),
            ("Bfactor.csv","Bfactor (Å)"),
        ]
        tif_file = 'Bfactor.tif'
        title = 'Bfactor of Solute Heavy Atoms'
        x_label = 'Residue No.'
        y_label = 'Bfactor (Å)'
        curve_label = ['Solute']
        color_list = ['#16499D']
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3,)      

    # 将结果转移至analysis-output目录;
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#-------------------------------系统性质计算"Rg"--------------------------------#
if args.rg or args.all or args.mode == '1' or args.mode == '2' or args.mode == '3':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "rg.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算Rg（Radius of Gyration）;
radgyr :{complex}&!(@H=) out Rg.txt mass nomax
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "rg.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算Rg（Radius of Gyration）;
radgyr :{solute}&!(@H=) out Rg.txt mass nomax
"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('rg.in')
    print("Rg计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    insert_delimiter_and_headers("Rg.txt", "Rg.csv", 1, ["Time", "Rg"])
  
    # 绘图(Rg.tif);
    csv_files = [
        ("Rg.csv","Time"),
        ("Rg.csv","Rg"),
    ]
    tif_file = 'Rg.tif'
    if args.receptor and args.ligand:
        title = 'Rg of Complex Heavy Atoms'
        curve_label = ['Complex']
    if args.solute:
        title = 'Rg of Solute Heavy Atoms'
        curve_label = ['Solute']
    x_label = f'Time (×{unit_for_time} ns)'
    y_label = 'Rg'
    color_list = ['#16499D']
    plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3,)

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#-------------------------------系统性质计算"SASA"------------------------------#
if args.sasa or args.all or args.mode == '1' or args.mode == '2':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "sasa.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算溶质表面积Surf 或 溶剂可及表面积SASA;
# 复合结构的:
surf :{complex} out Surf_Complex.txt
# 配体的:
surf :{ligand} out Surf_Ligand.txt
# 受体的:
surf :{receptor} out Surf_Receptor.txt
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "sasa.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算溶质表面积Surf 或 溶剂可及表面积SASA;
surf :{solute} out Surf.txt

"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('sasa.in')
    print("SASA计算完成;")
    sys.stdout.flush() 

    if args.receptor and args.ligand:
        # 整理cpptraj输出数据;
        insert_delimiter_and_headers("Surf_Complex.txt", "Surf_Complex.csv", 1, ["Time", "Surface Area (Å^2)"])
        insert_delimiter_and_headers("Surf_Receptor.txt", "Surf_Receptor.csv", 1, ["Time", "Surface Area (Å^2)"])
        insert_delimiter_and_headers("Surf_Ligand.txt", "Surf_Ligand.csv", 1, ["Time", "Surface Area (Å^2)"])

        # 绘图(Surf.tif);
        csv_files = [
            ("Surf_Complex.csv", "Time"),
            ("Surf_Complex.csv", "Surface Area (Å^2)"),
            ("Surf_Receptor.csv", "Surface Area (Å^2)"),
            ("Surf_Ligand.csv", "Surface Area (Å^2)"),
        ]
        tif_file = 'Surf.tif'
        title = 'Surface Area of Complex/Receptor/Ligand'
        x_label = f'Time (×{unit_for_time} ns)'
        y_label = f'Surface Area ($Å^{2}$)'
        curve_label = ['Complex','Receptor','Ligand']
        color_list = ['#16499D', '#E71F19', '#36AE37',]
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3,)
        
        # 绘图(Avg_Surf.tif);
        csv_files = [
            ("Surf_Complex.csv", "Surface Area (Å^2)"),
            ("Surf_Receptor.csv", "Surface Area (Å^2)"),
            ("Surf_Ligand.csv", "Surface Area (Å^2)"),
        ]
        tif_file = "Avg_Surf.tif"
        title = "Average Value of Surface Area"
        x_label = "Term" 
        y_label = f'Avg. Surface Area ($Å^{2}$)'
        x_ticks_labels = ['Complex','Receptor','Ligand']
        color_list=['#16499D', '#E71F19', '#36AE37',]
        plot_avg_bar(csv_files, tif_file, title, x_label, y_label, x_ticks_labels, color_list, top_margin=0.3, bottom_margin=0.3)

    if args.solute:
        # 整理cpptraj输出数据;
        insert_delimiter_and_headers("Surf.txt", "Surf.csv", 1, ["Time", "Surface Area (Å^2)"])

        # 绘图(Surf.tif);
        csv_files = [
            ("Surf.csv", "Time"),
            ("Surf.csv", "Surface Area (Å^2)"),
        ]
        tif_file = 'Surf.tif'
        title = 'Surface Area of Solute'
        x_label = f'Time (×{unit_for_time} ns)'
        y_label = f'Surface Area ($Å^{2}$)'
        curve_label = ['Solute']
        color_list = ['#16499D',]
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3,)
    
    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#-----------------------------ele+vdw相互作用能计算-----------------------------#
if args.nbi or args.all or args.mode == '1':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "nbi.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 用于确认配体周围残基（小于4.5埃）的参照结构;
reference cmp.inpcrd
# 计算分子间非键相互作用（静电和范德华）;
lie Energy ":{ligand}" ":{ligand}<:4.5 & !:WAT & !:Na+ & !:Cl- & !:{ligand}" out Non-Bonded_Interactions.txt
"""
            f.write(content)

        # 运行cpptraj程序
        run_cpptraj('nbi.in')
        print("ele+vdw相互作用能计算完成;")
        sys.stdout.flush()
 
        # 整理cpptraj输出数据;
        insert_delimiter_and_headers("Non-Bonded_Interactions.txt", "Non-Bonded_Interactions.csv", 1, ["Time", "Eele", "Evdw"])

        # 绘图(Non-Bonded_Interactions.tif);
        csv_files = [
            ("Non-Bonded_Interactions.csv", "Time"),
            ("Non-Bonded_Interactions.csv", "Eele"),
            ("Non-Bonded_Interactions.csv", "Evdw"),
        ]
        tif_file = 'Non-Bonded_Interactions.tif'
        title = 'Non-Bonded Interactions Energy Over Time'
        x_label = f'Time (×{unit_for_time} ns)'
        y_label = 'Non-Bonded Interactions Energy (kcal/mol)'
        curve_label = ['Eele','Evdw']
        color_list = ['#16499D', '#E71F19', ]
        plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3)

        # 将结果转移至analysis-output目录
        data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
        for file in data_output:
            shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#---------------------------------PCA主成分分析---------------------------------#
if args.pca or args.all or args.mode == '2':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "pca.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 构建一个'平均构象数据集'和'轨迹数据集';
average crdset average-dataset
createcrd MyTrajectories
run
# 以平均构象为参照，计算轨迹数据集的rmsd;
crdaction MyTrajectories rms ref average-dataset :{complex}@CA,C,N,
# 计算协方差矩阵;
crdaction MyTrajectories matrix covar name MyCovar :{complex}@CA,C,N,
# 获取前20个特征向量;
runanalysis diagmatrix MyCovar out Eigenvalue_Vector.txt vecs 20 name MyEvecs nmwiz nmwizvecs 20 nmwizfile Eigenvalue_Vector_VMD.nmd nmwizmask :{complex}@CA,C,N,
# 将主成分特征向量投影到轨迹中;
crdaction MyTrajectories projection PCA modes MyEvecs out Principal_Component.txt beg 1 end 20 :{complex}@CA,C,N, crdframes 1,last
# 绘制标准直方图;
hist PCA:1 bins 200 out Principal_Component_hist.agr norm name PC-1
hist PCA:2 bins 200 out Principal_Component_hist.agr norm name PC-2
hist PCA:3 bins 200 out Principal_Component_hist.agr norm name PC-3
run
# 创建沿第一第二个PC运动的NetCDF伪轨迹文件;
clear all
readdata Eigenvalue_Vector.txt name Evecs
parm cmp.prmtop
parmstrip !(:{complex}@CA,C,N,)
parmwrite out PCA-Mode1.prmtop
runanalysis modes name Evecs trajout PCA-Mode1.nc pcmin -200 pcmax 200 tmode 1 trajoutmask :{complex}&!@H= trajoutfmt netcdf
runanalysis modes name Evecs trajout PCA-Mode2.nc pcmin -200 pcmax 200 tmode 2 trajoutmask :{complex}&!@H= trajoutfmt netcdf
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "pca.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 构建一个'平均构象数据集'和'轨迹数据集';
average crdset average-dataset
createcrd MyTrajectories
run
# 以平均构象为参照，计算轨迹数据集的rmsd;
crdaction MyTrajectories rms ref average-dataset :{solute}@CA,C,N,
# 计算协方差矩阵;
crdaction MyTrajectories matrix covar name MyCovar :{solute}@CA,C,N,
# 获取前20个特征向量;
runanalysis diagmatrix MyCovar out Eigenvalue_Vector.txt vecs 20 name MyEvecs nmwiz nmwizvecs 20 nmwizfile Eigenvalue_Vector_VMD.nmd nmwizmask :{solute}@CA,C,N,
# 将主成分特征向量投影到轨迹中;
crdaction MyTrajectories projection PCA modes MyEvecs out Principal_Component.txt beg 1 end 20 :{solute}@CA,C,N, crdframes 1,last
# 绘制标准直方图;
hist PCA:1 bins 200 out Principal_Component_hist.agr norm name PC-1
hist PCA:2 bins 200 out Principal_Component_hist.agr norm name PC-2
hist PCA:3 bins 200 out Principal_Component_hist.agr norm name PC-3
# 创建沿第一第二个PC运动的NetCDF伪轨迹文件;
run
clear all
readdata Eigenvalue_Vector.txt name Evecs
parm cmp.prmtop
parmstrip !(:{solute}@CA,C,N,)
parmwrite out PCA-Mode1.prmtop
runanalysis modes name Evecs trajout PCA-Mode1.nc pcmin -200 pcmax 200 tmode 1 trajoutmask :{solute}&!@H= trajoutfmt netcdf
runanalysis modes name Evecs trajout PCA-Mode2.nc pcmin -200 pcmax 200 tmode 2 trajoutmask :{solute}&!@H= trajoutfmt netcdf
"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('pca.in')
    print("PCA主成分计算完成;")
    sys.stdout.flush()
 
    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") + glob.glob("*.agr") + glob.glob("*.nmd") + glob.glob("PCA-Mode*.*") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))
        

#-------------------------------系统性质计算"MSD"-------------------------------#
if args.msd or args.all or args.mode == '3': 
    msd_molecules = [item for sublist in args.msd_mol for item in sublist] # args.msd_mol是二维列表，将其展开为一维列表
    msd_molecules = list(set(msd_molecules))  # 移除重复元素
    print(f"计算{msd_molecules}分子/离子的MSD;")
    
    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;    
        Input_File_Path = os.path.join(Analysis_Input_Folder, "msd.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算MSD及扩散系数;
"""
            for msd_molecule in msd_molecules:
                content += f"diffusion :{msd_molecule} out MSD_{msd_molecule}.txt {msd_molecule} diffout Diffusion_Contants_{msd_molecule}.txt\n"
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;    
        Input_File_Path = os.path.join(Analysis_Input_Folder, "msd.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算MSD及扩散系数;
"""
            for msd_molecule in msd_molecules:
                content += f"diffusion :{msd_molecule} out MSD_{msd_molecule}.txt {msd_molecule} diffout Diffusion_Contants_{msd_molecule}.txt\n"
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('msd.in')
    print("MSD计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    for msd_molecule in msd_molecules:
        insert_delimiter_and_headers(f"MSD_{msd_molecule}.txt", f"MSD_{msd_molecule}.csv", 1, ["Time", f"MSD [{msd_molecule}] (Å^2)", "[X] (Å^2)", "[Y] (Å^2)", "[Z] (Å^2)", "[A] (Å^2)"])
        change_index_name(f"Diffusion_Contants_{msd_molecule}.txt", "Set")
        insert_delimiter_and_headers(f"Diffusion_Contants_{msd_molecule}.txt", f"Diffusion_Contants_{msd_molecule}.csv", 0, )
    
    # 绘图(MSD.tif);
    csv_files = []
    curve_label = []
    csv_files.append((f"MSD_{msd_molecule}.csv", "Time"))
    for msd_molecule in msd_molecules:
        csv_files.append((f"MSD_{msd_molecule}.csv", f"MSD [{msd_molecule}] (Å^2)"))
        curve_label.append(f"{msd_molecule}")
    tif_file = 'MSD.tif'
    title = f'MSD of {", ".join(msd_molecules)}'
    x_label = f'Time (×{unit_for_time} ns)'
    y_label = f'MSD ($Å^{2}$)'
    color_list = ['red', 'blue', 'green', 'yellow', 'purple', 'pink', 'orange', 'brown', 'gray']
    plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.05, bottom_margin=0.3)

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#-------------------------------系统性质计算"RDF"-------------------------------#
if args.rdf or args.all or args.mode == '3':

    if args.receptor and args.ligand:
        rdf_atm1 = args.rdf_atm1
        rdf_atm2 = args.rdf_atm2
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "rdf.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算RDF（Radial Distribution Function）;
radial RDF.txt 0.1 20.0 :{rdf_atm1} :{rdf_atm2} volume
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "rdf.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算RDF（Radial Distribution Function）;
radial RDF.txt 0.1 20.0 :{rdf_atm1} :{rdf_atm2} volume
"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('rdf.in')
    print("RDF计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    insert_delimiter_and_headers("RDF.txt", "RDF.csv", 1, ["r (Å)", "g(r)"])

    # 绘图(RDF.tif);
    csv_files = [
        ("RDF.csv","r (Å)"),
        ("RDF.csv","g(r)"),
    ]
    tif_file = 'RDF.tif'
    title = f'RDF of All Atoms From [:{rdf_atm1}] to Each Atom From [:{rdf_atm2}]'
    x_label = 'r (Å)'
    y_label = 'g(r)'
    curve_label = [f'g(r)[{rdf_atm1},{rdf_atm2}]']
    color_list = ['#16499D']
    plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.2, bottom_margin=0.3,)

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#-----------------------------系统性质计算"Density"-----------------------------#
if args.density or args.all or args.mode == '3':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "density.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 沿坐标计算体系密度
density out Density.txt number delta 0.25 ":{complex}"
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "density.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 沿坐标计算体系密度
density out Density.txt number delta 0.25 ":{solute}"
"""
            f.write(content)

    # 运行cpptraj程序;
    run_cpptraj('density.in')
    print("Density计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    insert_delimiter_and_headers("Density.txt", "Density.csv", 2, ["Distance (Å)", "Avg. Density (1/Å^3)", "Std Dev"])
     
    # 绘图(Density.tif);
    csv_files = [
        ("Density.csv","Distance (Å)"),
        ("Density.csv","Avg. Density (1/Å^3)"),
    ]
    tif_file = 'Density.tif'
    if args.receptor and args.ligand:
        title = 'Relative Distances vs. Densities for Complex'
        curve_label = ['Complex']
    if args.solute:
        title = 'Relative Distances vs. Densities for Solute'
        curve_label = ['Solute']
    x_label = 'Distance (Å)'
    y_label = f'Avg. Density (1/$Å^{3}%)'
    color_list = ['#16499D']
    plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3, plot_type='bar')

    # 将结果转移至analysis-output目录;
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#-----------------------------------氢键分析------------------------------------#
if args.hbond or args.all or args.mode == '1' or args.mode == '2' or args.mode == '3':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "hbond.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 用于确认配体周围残基（小于4.5埃）的参照结构;
reference cmp.inpcrd
# 氢键分析;
# 输出'溶质-溶质间氢键';
hbond HBOND-Solute angle 135 dist 3 \
donormask :{complex} acceptormask :{complex} \
avgout Hbond_Solute_Avg.txt \
series uuseries Hbond_Solute_TimeSeries.txt
go
# 输出'溶质-溶剂间氢键、溶质-溶剂-溶质间氢键（水桥）';
hbond HBOND-Solvent angle 135 dist 3 \
donormask :{complex} acceptormask :{complex} \
solventdonor :WAT solventacceptor :WAT@O \
solvout Hbond_Solvent_Avg.txt \
bridgeout Hbond_Bridge_Avg.txt \
series uvseries Hbond_Solvent_TimeSeries.txt
go
# 输出'溶质-溶质间氢键、溶质-溶剂间氢键、溶质-溶剂-溶质间氢键（水桥）'数量vs时间数据;
create Hbond_vs_Time.txt HBOND-Solute[UU] HBOND-Solvent[UV] HBOND-Solvent[Bridge]
go
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "hbond.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 氢键分析;
# 输出'溶质-溶质间氢键';
hbond HBOND-Solute angle 135 dist 3 \
donormask :{solute} acceptormask :{solute} \
avgout Hbond_Solute_Avg.txt \
series uuseries Hbond_Solute_TimeSeries.txt
go
# 输出'溶质-溶剂间氢键、溶质-溶剂-溶质间氢键（水桥）';
hbond HBOND-Solvent angle 135 dist 3 \
donormask :{solute} acceptormask :{solute} \
solventdonor :WAT solventacceptor :WAT@O \
solvout Hbond_Solvent_Avg.txt \
bridgeout Hbond_Bridge_Avg.txt \
series uvseries Hbond_Solvent_TimeSeries.txt
go
# 输出'溶质-溶质间氢键、溶质-溶剂间氢键、溶质-溶剂-溶质间氢键（水桥）'数量vs时间数据;
create Hbond_vs_Time.txt HBOND-Solute[UU] HBOND-Solvent[UV] HBOND-Solvent[Bridge]
go
"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('hbond.in')
    print("氢键计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    insert_delimiter_and_headers("Hbond_vs_Time.txt", "Hbond_vs_Time.csv", 1, ["Time", "HBOND-Solute[UU]", "HBOND-Solvent[UV]", "HBOND-Solvent[Bridge]"])
    insert_delimiter_and_headers("Hbond_Solute_Avg.txt", "Hbond_Solute_Avg.csv", 0, )
    insert_delimiter_and_headers("Hbond_Solvent_Avg.txt", "Hbond_Solvent_Avg.csv", 0, )
    insert_delimiter_and_headers("Hbond_Bridge_Avg.txt", "Hbond_Bridge_Avg.csv", 0, )
    change_index_name("Hbond_Solute_TimeSeries.txt", "Time")
    change_index_name("Hbond_Solvent_TimeSeries.txt", "Time")
    insert_delimiter_and_headers("Hbond_Solute_TimeSeries.txt", "Hbond_Solute_TimeSeries.csv", 0, )
    insert_delimiter_and_headers("Hbond_Solvent_TimeSeries.txt", "Hbond_Solvent_TimeSeries.csv", 0, )

    # 绘图(Hbond_vs_Time.tif);
    csv_files = [
        ("Hbond_vs_Time.csv", "Time"),
        ("Hbond_vs_Time.csv", "HBOND-Solute[UU]"),
        ("Hbond_vs_Time.csv", "HBOND-Solvent[UV]"),
        ("Hbond_vs_Time.csv", "HBOND-Solvent[Bridge]"),
    ]
    tif_file = 'Hbond_vs_Time.tif'
    title = 'Number of Hydrogen Bonds Over Time'
    x_label = f'Time (×{unit_for_time} ns)'
    y_label = 'Number of Hydrogen Bonds'
    curve_label = ['Solute[UU]','Solvent[UV]','Solvent[Bridge]']
    color_list = ['#16499D', '#E71F19', '#36AE37', ]
    plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3)

    # 绘图(Avg_Hydrogen_Bonds.tif);
    csv_files = [
        ("Hbond_vs_Time.csv", "HBOND-Solute[UU]"),
        ("Hbond_vs_Time.csv", "HBOND-Solvent[UV]"),
        ("Hbond_vs_Time.csv", "HBOND-Solvent[Bridge]"),
    ]
    tif_file = "Avg_Hydrogen_Bonds.tif"
    title = "Average Number of Hydrogen Bonds"
    x_label = "Term" 
    y_label = "Number of Hydrogen Bonds"
    x_ticks_labels = ['Solute[UU]','Solvent[UV]','Solvent[Bridge]']
    color_list=['#16499D', '#E71F19', '#36AE37', ]
    plot_avg_bar(csv_files, tif_file, title, x_label, y_label, x_ticks_labels, color_list, top_margin=0.3, bottom_margin=0.3, int_yvalue=True)

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#---------------------------------二级结构分析----------------------------------#
if args.dssp or args.all or args.mode == '2':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "dssp.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 分析二级结构;
# 独立算受体二级结构;
secstruct :{receptor} out Dssp.txt sumout Dssp_Sum.txt assignout Dssp_Assign.txt totalout Dssp_Total.txt
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "dssp.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 分析二级结构;
secstruct :{solute} out Dssp.txt sumout Dssp_Sum.txt assignout Dssp_Assign.txt totalout Dssp_Total.txt
"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('dssp.in')
    print("二级结构计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    insert_delimiter_and_headers("Dssp_Sum.txt", "Dssp_Sum.csv", 1, ["Residue No.", "Para", "Anti", "3-10", "Alpha", "Pi", "Turn", "Bend"])
    insert_delimiter_and_headers("Dssp_Total.txt", "Dssp_Total.csv", 1, ["Time", "Para", "Anti", "3-10", "Alpha", "Pi", "Turn", "Bend"])
    change_index_name("Dssp.txt", "Time")
    insert_delimiter_and_headers("Dssp.txt", "Dssp.csv", 0, )
    
    # 绘图(Dssp.tif);
    csv_file = 'Dssp.csv'
    tif_file = 'Dssp.tif'
    title = 'Secondary Structure for Each Residue Over Time'
    x_label = f'Time (×{unit_for_time} ns)'
    y_label = 'Residue No.'
    colorbar_label = 'Secondary Structure Type'
    plot_matrix_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label)

    # 绘图(Dssp_Sum.tif);
    csv_file = 'Dssp_Sum.csv'
    tif_file = 'Dssp_Sum.tif'
    title = 'Average Structural Propensities for Each Residue'
    x_label = 'Residue No.'
    y_label = 'Secondary Structure Type'
    colorbar_label = 'Structural Propensity'
    colorbar_cmap = 'Blues'
    plot_matrix_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label, colorbar_cmap)

    # 绘图(Dssp_Total.tif);
    csv_file = 'Dssp_Total.csv'
    tif_file = 'Dssp_Total.tif'
    title = 'Total Structural Propensity for All Residues Over Time'
    x_label = f'Time (×{unit_for_time} ns)'
    y_label = 'Secondary Structure Type'
    colorbar_label = 'Structural Propensity'
    colorbar_cmap = 'Reds'
    plot_matrix_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label, colorbar_cmap)

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#--------------------------------残基接触矩阵计算-------------------------------#
if args.contact_matrix or args.all or args.mode == '1' or args.mode == '2':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "contact_matrix.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算残基接触矩阵; 
reference cmp.inpcrd
nativecontacts name CONTACT :{complex} \
byresidue distance 4.5 reference \
out Contacts_vs_Time.txt mindist maxdist \
writecontacts Contacts_Frac_ByAtom.txt \
resout Contacts_Frac_ByRes.txt \
map mapout Contacts_Map.txt \
contactpdb Contacts.pdb \
series seriesout Contacts_Series.txt
run
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "contact_matrix.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 计算残基接触矩阵; 
reference cmp.inpcrd
nativecontacts name CONTACT :{solute} \
byresidue distance 4.5 reference \
out Contacts_vs_Time.txt mindist maxdist \
writecontacts Contacts_Frac_ByAtom.txt \
resout Contacts_Frac_ByRes.txt \
map mapout Contacts_Map.txt \
contactpdb Contacts.pdb \
series seriesout Contacts_Series.txt
run
"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('contact_matrix.in')
    print("残基接触矩阵计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    insert_delimiter_and_headers("Contacts_vs_Time.txt", "Contacts_vs_Time.csv", 1, ["Time", "Native Contacts", "Non-Native Contacts", "Min Dist. (Å)", "Max Dist. (Å)"])
    insert_delimiter_and_headers("Contacts_Frac_ByAtom.txt", "Contacts_Frac_ByAtom.csv", 3, ["Set", "Contact", "Nframes", "Frac", "Avg", "Stdev"])
    insert_delimiter_and_headers("Contacts_Frac_ByRes.txt", "Contacts_Frac_ByRes.csv", 1, ["Residue No.", "Residue No.", "TotalFrac", "Number of Contacts"])
    insert_delimiter_and_headers("native.Contacts_Map.txt", "Native_Contacts_Map.csv", 1, ["Residue No.", "Residue No.", "Frac"])
    insert_delimiter_and_headers("nonnative.Contacts_Map.txt", "Non-native_Contacts_Map.csv", 1, ["Residue No.", "Residue No.", "Frac"])

    # 绘图(Contacts_vs_Time.tif);
    csv_files = [
        ("Contacts_vs_Time.csv", "Time"),
        ("Contacts_vs_Time.csv", "Native Contacts"),
        ("Contacts_vs_Time.csv", "Non-Native Contacts"),
    ]
    tif_file = 'Contacts_vs_Time.tif'
    title = 'Number of Native/Non-Native Contacts Over Time'
    x_label = f'Time (×{unit_for_time} ns)'
    y_label = 'Number of Contacts'
    curve_label = ['Native', 'Non-Native']
    color_list = ['red', 'blue', ]
    plot_curve_or_bar(csv_files, tif_file, title, x_label, y_label, curve_label, color_list, top_margin=0.3, bottom_margin=0.3)

    # 绘图(Avg_Contacts.tif);
    csv_files = [
        ("Contacts_vs_Time.csv", "Native Contacts"),
        ("Contacts_vs_Time.csv", "Non-Native Contacts")
    ]
    tif_file = "Avg_Contacts.tif"
    title = "Average Number of Native/Non-Native Contacts"
    x_label = "Term" 
    y_label = "Avg. Number of Contacts"
    x_ticks_labels = ['Native', 'Non-Native']
    color_list=['red', 'blue', ]
    plot_avg_bar(csv_files, tif_file, title, x_label, y_label, x_ticks_labels, color_list, top_margin=0.3, bottom_margin=0.3)

    # 绘图(Native_Contacts_Map.tif);
    csv_file = 'Native_Contacts_Map.csv'
    tif_file = 'Native_Contacts_Map.tif'
    title = 'Native Contacts Map'
    x_label = 'Residue No.'
    y_label = 'Residue No.'
    colorbar_label = 'Frequency'
    colorbar_cmap = 'Reds'       # tab10 tab20 tab20b tab20c 
    plot_xyz_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label, colorbar_cmap, x=0, y=1, z=2, )

    # 绘图(Non-native_Contacts_Map.tif);
    csv_file = 'Non-native_Contacts_Map.csv'
    tif_file = 'Non-native_Contacts_Map.tif'
    title = 'Non-native Contacts Map'
    x_label = 'Residue No.'
    y_label = 'Residue No.'
    colorbar_label = 'Frequency'
    colorbar_cmap = 'Blues'       # tab10 tab20 tab20b tab20c 
    plot_xyz_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label, colorbar_cmap, x=0, y=1, z=2, )

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") + glob.glob("Contacts.pdb") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#------------------------------动态相关性矩阵计算-------------------------------#
if args.correlation_matrix or args.all or args.mode == '2':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "correlation_matrix.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 构建一个'平均构象数据集'和'轨迹数据集';
average crdset average-dataset
createcrd MyTrajectories
run
# 以平均构象为参照，计算轨迹数据集的rmsd;
crdaction MyTrajectories rms ref average-dataset :{complex}@CA
# 计算协方差矩阵;
crdaction MyTrajectories matrix correl out Correlation_Matrix_CA.txt name CA_matrix :{complex}@CA
run
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "correlation_matrix.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 构建一个'平均构象数据集'和'轨迹数据集';
average crdset average-dataset
createcrd MyTrajectories
run
# 以平均构象为参照，计算轨迹数据集的rmsd;
crdaction MyTrajectories rms ref average-dataset :{solute}@CA
# 计算协方差矩阵;
crdaction MyTrajectories matrix correl out Correlation_Matrix_CA.txt name CA_matrix :{solute}@CA
run
"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('correlation_matrix.in')
    print("动态相关性矩阵计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    insert_delimiter_and_headers("Correlation_Matrix_CA.txt", "Correlation_Matrix_CA.csv", 0, )
    data = pd.read_csv("Correlation_Matrix_CA.csv", index_col=None, header=None)
    data.columns  = [np.arange(1, len(data.columns) + 1)]
    data.index = [np.arange(1, len(data.index) + 1)]
    data.to_csv("Correlation_Matrix_CA.csv", index=True, header=True)

    # 绘图(Correlation_Matrix_CA.tif);
    csv_file = 'Correlation_Matrix_CA.csv'
    tif_file = 'Correlation_Matrix_CA.tif'
    title = 'Dynamic Cross-Correlation Matrix'
    x_label = 'Residue No.'
    y_label = 'Residue No.'
    colorbar_label = 'Correlation'
    colorbar_cmap = 'seismic'
    plot_matrix_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label, colorbar_cmap, transpose_matrix=False )

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))


#---------------------------------帧间RMSD计算----------------------------------#
if args.rmsd_2d or args.all or args.mode == '2':

    if args.receptor and args.ligand:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "2d_rmsd.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 帧间RMSD; 
rms2d mass :{complex}&!(@H=) out 2D-RMSD.txt
run
"""
            f.write(content)

    if args.solute:
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "2d_rmsd.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入轨迹文件mdcrd;
{"".join(trajectory_files)}
# 居中模拟轨迹;
autoimage
# 帧间RMSD; 
rms2d mass :{solute}&!(@H=) out 2D-RMSD.txt
run
"""
            f.write(content)

    # 运行cpptraj程序
    run_cpptraj('2d_rmsd.in')
    print("帧间RMSD计算完成;")
    sys.stdout.flush()

    # 整理cpptraj输出数据;
    change_index_name("2D-RMSD.txt", "Time")
    insert_delimiter_and_headers("2D-RMSD.txt", "2D-RMSD.csv", 0, )
    
    # 绘图(2D-RMSD.tif);
    csv_file = '2D-RMSD.csv'
    tif_file = '2D-RMSD.tif'
    title = 'Inter-frame RMSD matrix'
    x_label = f'Time (×{unit_for_time} ns)'
    y_label = f'Time (×{unit_for_time} ns)'
    colorbar_label = 'RMSD (Å)'
    colorbar_cmap = 'Purples'
    plot_matrix_heatmap(csv_file, tif_file, title, x_label, y_label, colorbar_label, colorbar_cmap, transpose_matrix=False )

    # 将结果转移至analysis-output目录
    data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
    for file in data_output:
        shutil.move(file, os.path.join(Analysis_Output_Folder, file))

    
#-----------------------------MM/GBSA结合自由能计算-----------------------------#
if args.mmgbsa or args.all or args.mode == '1':

    # 必须定义receptor和ligand;     
    if args.receptor and args.ligand:  
        # 编写cpptraj程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "image.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
# 载入拓扑文件prmtop;
parm cmp.prmtop
# 载入平衡段轨迹文件mdcrd;
{"".join(equilibrium_segment_trajectory_files)}
# 居中模拟轨迹;
autoimage
# 输出平衡段模拟轨迹;
trajout cmp_eq{eq_start}-{eq_end}_image.mdcrd
"""
            f.write(content)
        
        # 运行cpptraj程序;
        run_cpptraj('image.in')
    
        # 运行cpptraj程序确定平衡段轨迹帧数;
        command_for_determine_farme = "cpptraj -p cmp.prmtop -y cmp_eq{}-{}_image.mdcrd -tl".format(eq_start, eq_end)
        result = subprocess.run(command_for_determine_farme, shell=True, capture_output=True, text=True)
        image_frame = result.stdout
        frames_used_for_mmpbsa_calculations = image_frame.split()[1]
        print(f'为了计算结合自由能，提取了平衡段轨迹，区间为{eq_start}-{eq_end} ns，共{frames_used_for_mmpbsa_calculations}帧;')    
    
        # 运行ante-MMPBSA.py程序，准备com.prmtop(真空相-复合结构)、rec.prmtop(真空相-受体结构)、lig.prmtop(真空相-配体结构)拓扑文件;
        run_ante_mmpbsa()
    
        # 编写MMPBSA程序输入/控制文件;
        Input_File_Path = os.path.join(Analysis_Input_Folder, "mmgbsa.in")
        with open(Input_File_Path, "w") as f:
            content = f"""
&general
startframe=1, endframe={frames_used_for_mmpbsa_calculations}, interval=1,
verbose=2, keep_files=2,
/
&gb
igb=8, saltcon=0.15,
/
"""
            # 如果要算残基能量分解数据，追加decomp设置;
            if args.decomp or args.all or args.mode == '1':
                content += f"""
&decomp
idecomp=2, dec_verbose=2,
/
"""     
            f.write(content)
        
        # 运行MMPBSA.py程序
        run_mmpbsa('mmgbsa.in')
        print("结合自由能计算完成;")
        sys.stdout.flush()
        
        # 整理输出数据;
        insert_delimiter_and_headers("MMGBSA_BindingEnergy.txt", "MMGBSA_BindingEnergy.csv", 0, )
        if args.decomp:
            insert_delimiter_and_headers("MMGBSA_EnergyDecomp.txt", "MMGBSA_EnergyDecomp.csv", 0, )

        # 将结果转移至analysis-output目录
        data_output = [file for file in glob.glob("*.txt") if file != 'config.txt']
        for file in data_output:
            shutil.move(file, os.path.join(Analysis_Output_Folder, file))
    
        # 将结果转移至analysis-log目录
        data_output = glob.glob("_MMPBSA*")
        for file in data_output:
            shutil.move(file, os.path.join(Analysis_Log_Folder, file))


#------------------------------------重新运行----------------------------------#
if args.restart:
    # 创建一个analysis-output-restart新目录用于保存Restart的输出文件;
    Analysis_Output_Folder_Restart = "analysis-output-restart"
    os.makedirs(Analysis_Output_Folder_Restart, exist_ok=True)

    # 读取Analysis_Input_Folder中的所有"*.in"输入文件;
    input_files = glob.glob(os.path.join(Analysis_Input_Folder, "*.in"))
    for input_file in input_files:
        input_filename = os.path.basename(input_file)

        if input_filename == 'mmgbsa.in':
            # 运行ante-MMPBSA.py程序;
            run_ante_mmpbsa()
            # 运行MMPBSA.py程序;
            run_mmpbsa('mmgbsa.in')
            print(f"MMPBSA.py程序运行了 {input_filename};")
            sys.stdout.flush()
        else:
            # 运行CPPTRAJ程序;
            run_cpptraj(input_filename)
            print(f"CPPTRAJ程序运行了 {input_filename};")
            sys.stdout.flush()

        # 将结果转移至analysis-output-restart新目录
        data_output = [file for file in glob.glob("*.txt") + glob.glob("*.agr") + glob.glob("*.nmd") + glob.glob("Avg_structure.pdb") + glob.glob("Contacts.pdb") if file != 'config.txt']
        for file in data_output:
            shutil.move(file, os.path.join(Analysis_Output_Folder_Restart, file))

        # 将结果转移至analysis-log目录
        data_output = glob.glob("_MMPBSA*")
        for file in data_output:
            shutil.move(file, os.path.join(Analysis_Log_Folder, file))


#------------------------------------------------------------------------------#
