#-*- coding: utf-8 -*-
"""
@author: TOYOBO CO., LTD.
"""
# Import functions
import pandas as pd
import numpy as np
import os
import pathlib
import shutil
import json
import math
import pubchempy as pcp
from rdkit import rdBase, Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole
from ipywidgets import interact, fixed, IntSlider
from tqdm import tqdm
from tqdm.notebook import tqdm
from tybmilib import prep
from tybmilib import myfilename as mfn
from tybmilib import logmgmt
import skimage.io
import skimage.util
from PIL import Image, ImageFilter

#------------------------------------------------------------
Local_mode = mfn.get_localmode()
DrawingOptions.bondLineWidth = 5

class Lib_ParseError(Exception):
    """module内エラー出力用のクラス
    
    モジュール内で発生した固有の処理エラーに対し、指定のExceptionクラスを付与し、出力をするためのクラス
    """
    pass

# 分子構造画像の連結用の関数
def get_concat_h_blank(im1, im2, color=(255, 255, 255)):
    dst = Image.new('RGB', (im1.width + im2.width, max(im1.height, im2.height)), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v_blank(im1, im2, color=(255, 255, 255)):
    dst = Image.new('RGB', (max(im1.width, im2.width), im1.height + im2.height), color)
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst
    
def get_concat_h_multi_blank(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_h_blank(_im, im)
    return _im

def get_concat_v_multi_blank(im_list):
    _im = im_list.pop(0)
    for im in im_list:
        _im = get_concat_v_blank(_im, im)
    return _im

def get_concat_tile_blank(im_list_2d):
    im_list_v = [get_concat_h_multi_blank(im_list_h) for im_list_h in im_list_2d]
    return get_concat_v_multi_blank(im_list_v)
###ここまで###


class Features:
    
    def __init__(self, master_path, df_columns, experiment_ID, structure_mode="", radius=0, bit_num=0):
        """コンストラクタ
        Args:
            master_path (str): 1st argument
            experiment_ID (str): 2nd argument
            structure_mode (str): 3rd argument
            radius (int): 4th argument
            bit_num (int): 5th argument
        Returns:
            None

        """
        self.experiment_ID = experiment_ID
        source_list = []
        if master_path != "" and master_path != None:
            self.df_master = prep.read_s3_bucket_data(master_path,experiment_ID)
            for source_name in df_columns:
                if source_name in list(self.df_master["Source_Name"]):
                    source_list.append(source_name)
        else:
            self.df_master = pd.DataFrame()

        self.source_list = source_list
        self.structure_mode=structure_mode
        self.radius=radius
        self.bit_num=bit_num
        self.mol_list = []
        self.duplicate_check = False

        self.SMARTS_PATTERNS = {
            1: ('?', 0),  # ISOTOPE
            2: ('[#104]', 0),  # limit the above def'n since the RDKit only accepts up to #104
            3: ('[#32,#33,#34,#50,#51,#52,#82,#83,#84]', 0),  # Group IVa,Va,VIa Rows 4-6 
            4: ('[Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr]', 0),  # actinide
            5: ('[Sc,Ti,Y,Zr,Hf]', 0),  # Group IIIB,IVB (Sc...)  
            6: ('[La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu]', 0),  # Lanthanide
            7: ('[V,Cr,Mn,Nb,Mo,Tc,Ta,W,Re]', 0),  # Group VB,VIB,VIIB
            8: ('[!#6;!#1]1~*~*~*~1', 0),  # QAAA@1
            9: ('[Fe,Co,Ni,Ru,Rh,Pd,Os,Ir,Pt]', 0),  # Group VIII (Fe...)
            10: ('[Be,Mg,Ca,Sr,Ba,Ra]', 0),  # Group IIa (Alkaline earth)
            11: ('*1~*~*~*~1', 0),  # 4M Ring
            12: ('[Cu,Zn,Ag,Cd,Au,Hg]', 0),  # Group IB,IIB (Cu..)
            13: ('[#8]~[#7](~[#6])~[#6]', 0),  # ON(C)C
            14: ('[#16]-[#16]', 0),  # S-S
            15: ('[#8]~[#6](~[#8])~[#8]', 0),  # OC(O)O
            16: ('[!#6;!#1]1~*~*~1', 0),  # QAA@1
            17: ('[#6]#[#6]', 0),  #CTC
            18: ('[#5,#13,#31,#49,#81]', 0),  # Group IIIA (B...) 
            19: ('*1~*~*~*~*~*~*~1', 0),  # 7M Ring
            20: ('[#14]', 0),  #Si
            21: ('[#6]=[#6](~[!#6;!#1])~[!#6;!#1]', 0),  # C=C(Q)Q
            22: ('*1~*~*~1', 0),  # 3M Ring
            23: ('[#7]~[#6](~[#8])~[#8]', 0),  # NC(O)O
            24: ('[#7]-[#8]', 0),  # N-O
            25: ('[#7]~[#6](~[#7])~[#7]', 0),  # NC(N)N
            26: ('[#6]=;@[#6](@*)@*', 0),  # C$=C($A)$A
            27: ('[I]', 0),  # I
            28: ('[!#6;!#1]~[CH2]~[!#6;!#1]', 0),  # QCH2Q
            29: ('[#15]', 0),  # P
            30: ('[#6]~[!#6;!#1](~[#6])(~[#6])~*', 0),  # CQ(C)(C)A
            31: ('[!#6;!#1]~[F,Cl,Br,I]', 0),  # QX
            32: ('[#6]~[#16]~[#7]', 0),  # CSN
            33: ('[#7]~[#16]', 0),  # NS
            34: ('[CH2]=*', 0),  # CH2=A
            35: ('[Li,Na,K,Rb,Cs,Fr]', 0),  # Group IA (Alkali Metal)
            36: ('[#16R]', 0),  # S Heterocycle
            37: ('[#7]~[#6](~[#8])~[#7]', 0),  # NC(O)N
            38: ('[#7]~[#6](~[#6])~[#7]', 0),  # NC(C)N
            39: ('[#8]~[#16](~[#8])~[#8]', 0),  # OS(O)O
            40: ('[#16]-[#8]', 0),  # S-O
            41: ('[#6]#[#7]', 0),  # CTN
            42: ('F', 0),  # F
            43: ('[!#6;!#1;!H0]~*~[!#6;!#1;!H0]', 0),  # QHAQH
            44: ('[!#1;!#6;!#7;!#8;!#9;!#14;!#15;!#16;!#17;!#35;!#53]', 0),  # OTHER
            45: ('[#6]=[#6]~[#7]', 0),  # C=CN
            46: ('Br', 0),  # BR
            47: ('[#16]~*~[#7]', 0),  # SAN
            48: ('[#8]~[!#6;!#1](~[#8])(~[#8])', 0),  # OQ(O)O
            49: ('[!+0]', 0),  # CHARGE  
            50: ('[#6]=[#6](~[#6])~[#6]', 0),  # C=C(C)C
            51: ('[#6]~[#16]~[#8]', 0),  # CSO
            52: ('[#7]~[#7]', 0),  # NN
            53: ('[!#6;!#1;!H0]~*~*~*~[!#6;!#1;!H0]', 0),  # QHAAAQH
            54: ('[!#6;!#1;!H0]~*~*~[!#6;!#1;!H0]', 0),  # QHAAQH
            55: ('[#8]~[#16]~[#8]', 0),  #OSO
            56: ('[#8]~[#7](~[#8])~[#6]', 0),  # ON(O)C
            57: ('[#8R]', 0),  # O Heterocycle
            58: ('[!#6;!#1]~[#16]~[!#6;!#1]', 0),  # QSQ
            59: ('[#16]!:*:*', 0),  # Snot%A%A
            60: ('[#16]=[#8]', 0),  # S=O
            61: ('*~[#16](~*)~*', 0),  # AS(A)A
            62: ('*@*!@*@*', 0),  # A$!A$A
            63: ('[#7]=[#8]', 0),  # N=O
            64: ('*@*!@[#16]', 0),  # A$A!S
            65: ('c:n', 0),  # C%N
            66: ('[#6]~[#6](~[#6])(~[#6])~*', 0),  # CC(C)(C)A
            67: ('[!#6;!#1]~[#16]', 0),  # QS
            68: ('[!#6;!#1;!H0]~[!#6;!#1;!H0]', 0),  # QHQH (&...) SPEC Incomplete
            69: ('[!#6;!#1]~[!#6;!#1;!H0]', 0),  # QQH
            70: ('[!#6;!#1]~[#7]~[!#6;!#1]', 0),  # QNQ
            71: ('[#7]~[#8]', 0),  # NO
            72: ('[#8]~*~*~[#8]', 0),  # OAAO
            73: ('[#16]=*', 0),  # S=A
            74: ('[CH3]~*~[CH3]', 0),  # CH3ACH3
            75: ('*!@[#7]@*', 0),  # A!N$A
            76: ('[#6]=[#6](~*)~*', 0),  # C=C(A)A
            77: ('[#7]~*~[#7]', 0),  # NAN
            78: ('[#6]=[#7]', 0),  # C=N
            79: ('[#7]~*~*~[#7]', 0),  # NAAN
            80: ('[#7]~*~*~*~[#7]', 0),  # NAAAN
            81: ('[#16]~*(~*)~*', 0),  # SA(A)A
            82: ('*~[CH2]~[!#6;!#1;!H0]', 0),  # ACH2QH
            83: ('[!#6;!#1]1~*~*~*~*~1', 0),  # QAAAA@1
            84: ('[NH2]', 0),  #NH2
            85: ('[#6]~[#7](~[#6])~[#6]', 0),  # CN(C)C
            86: ('[C;H2,H3][!#6;!#1][C;H2,H3]', 0),  # CH2QCH2
            87: ('[F,Cl,Br,I]!@*@*', 0),  # X!A$A
            88: ('[#16]', 0),  # S
            89: ('[#8]~*~*~*~[#8]', 0),  # OAAAO
            90:
                  ('[$([!#6;!#1;!H0]~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[CH2;R]1)]',
                   0),  # QHAACH2A
            91:
                  ('[$([!#6;!#1;!H0]~*~*~*~[CH2]~*),$([!#6;!#1;!H0;R]1@[R]@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~[R]1@[R]@[R]@[CH2;R]1),$([!#6;!#1;!H0]~*~[R]1@[R]@[CH2;R]1)]',
                   0),  # QHAAACH2A
            92: ('[#8]~[#6](~[#7])~[#6]', 0),  # OC(N)C
            93: ('[!#6;!#1]~[CH3]', 0),  # QCH3
            94: ('[!#6;!#1]~[#7]', 0),  # QN
            95: ('[#7]~*~*~[#8]', 0),  # NAAO
            96: ('*1~*~*~*~*~1', 0),  # 5 M ring
            97: ('[#7]~*~*~*~[#8]', 0),  # NAAAO
            98: ('[!#6;!#1]1~*~*~*~*~*~1', 0),  # QAAAAA@1
            99: ('[#6]=[#6]', 0),  # C=C
            100: ('*~[CH2]~[#7]', 0),  # ACH2N
            101:
                  ('[$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1),$([R]@1@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]@[R]1)]',
                   0),  # 8M Ring or larger. This only handles up to ring sizes of 14
            102: ('[!#6;!#1]~[#8]', 0),  # QO
            103: ('Cl', 0),  # CL
            104: ('[!#6;!#1;!H0]~*~[CH2]~*', 0),  # QHACH2A
            105: ('*@*(@*)@*', 0),  # A$A($A)$A
            106: ('[!#6;!#1]~*(~[!#6;!#1])~[!#6;!#1]', 0),  # QA(Q)Q
            107: ('[F,Cl,Br,I]~*(~*)~*', 0),  # XA(A)A
            108: ('[CH3]~*~*~*~[CH2]~*', 0),  # CH3AAACH2A
            109: ('*~[CH2]~[#8]', 0),  # ACH2O
            110: ('[#7]~[#6]~[#8]', 0),  # NCO
            111: ('[#7]~*~[CH2]~*', 0),  # NACH2A
            112: ('*~*(~*)(~*)~*', 0),  # AA(A)(A)A
            113: ('[#8]!:*:*', 0),  # Onot%A%A
            114: ('[CH3]~[CH2]~*', 0),  # CH3CH2A
            115: ('[CH3]~*~[CH2]~*', 0),  # CH3ACH2A
            116: ('[$([CH3]~*~*~[CH2]~*),$([CH3]~*1~*~[CH2]1)]', 0),  # CH3AACH2A
            117: ('[#7]~*~[#8]', 0),  # NAO
            118: ('[$(*~[CH2]~[CH2]~*),$(*1~[CH2]~[CH2]1)]', 1),  # ACH2CH2A > 1
            119: ('[#7]=*', 0),  # N=A
            120: ('[!#6;R]', 1),  # Heterocyclic atom > 1 (&...) Spec Incomplete
            121: ('[#7;R]', 0),  # N Heterocycle
            122: ('*~[#7](~*)~*', 0),  # AN(A)A
            123: ('[#8]~[#6]~[#8]', 0),  # OCO
            124: ('[!#6;!#1]~[!#6;!#1]', 0),  # QQ
            125: ('?', 0),  # Aromatic Ring > 1
            126: ('*!@[#8]!@*', 0),  # A!O!A
            127: ('*@*!@[#8]', 1),  # A$A!O > 1 (&...) Spec Incomplete
            128:
                  ('[$(*~[CH2]~*~*~*~[CH2]~*),$([R]1@[CH2;R]@[R]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[R]@[CH2;R]1),$(*~[CH2]~*~[R]1@[R]@[CH2;R]1)]',
                   0),  # ACH2AAACH2A
                129: ('[$(*~[CH2]~*~*~[CH2]~*),$([R]1@[CH2]@[R]@[R]@[CH2;R]1),$(*~[CH2]~[R]1@[R]@[CH2;R]1)]',
                        0),  # ACH2AACH2A
            130: ('[!#6;!#1]~[!#6;!#1]', 1),  # QQ > 1 (&...)  Spec Incomplete
            131: ('[!#6;!#1;!H0]', 1),  # QH > 1
            132: ('[#8]~*~[CH2]~*', 0),  # OACH2A
            133: ('*@*!@[#7]', 0),  # A$A!N
            134: ('[F,Cl,Br,I]', 0),  # X (HALOGEN)
            135: ('[#7]!:*:*', 0),  # Nnot%A%A
            136: ('[#8]=*', 1),  # O=A>1 
            137: ('[!C;!c;R]', 0),  # Heterocycle
            138: ('[!#6;!#1]~[CH2]~*', 1),  # QCH2A>1 (&...) Spec Incomplete
            139: ('[O;!H0]', 0),  # OH
            140: ('[#8]', 3),  # O > 3 (&...) Spec Incomplete
            141: ('[CH3]', 2),  # CH3 > 2  (&...) Spec Incomplete
            142: ('[#7]', 1),  # N > 1
            143: ('*@*!@[#8]', 0),  # A$A!O
            144: ('*!:*:*!:*', 0),  # Anot%A%Anot%A
            145: ('*1~*~*~*~*~*~1', 1),  # 6M ring > 1
            146: ('[#8]', 2),  # O > 2
            147: ('[$(*~[CH2]~[CH2]~*),$([R]1@[CH2;R]@[CH2;R]1)]', 0),  # ACH2CH2A
            148: ('*~[!#6;!#1](~*)~*', 0),  # AQ(A)A
            149: ('[C;H3,H4]', 1),  # CH3 > 1
            150: ('*!@*@*!@*', 0),  # A!A$A!A
            151: ('[#7;!H0]', 0),  # NH
            152: ('[#8]~[#6](~[#6])~[#6]', 0),  # OC(C)C
            153: ('[!#6;!#1]~[CH2]~*', 0),  # QCH2A
            154: ('[#6]=[#8]', 0),  # C=O
            155: ('*!@[CH2]!@*', 0),  # A!CH2!A
            156: ('[#7]~*(~*)~*', 0),  # NA(A)A
            157: ('[#6]-[#8]', 0),  # C-O
            158: ('[#6]-[#7]', 0),  # C-N
            159: ('[#8]', 1),  # O>1
            160: ('[C;H3,H4]', 0),  #CH3
            161: ('[#7]', 0),  # N
            162: ('a', 0),  # Aromatic
            163: ('*1~*~*~*~*~*~1', 0),  # 6M Ring
            164: ('[#8]', 0),  # O
            165: ('[R]', 0),  # Ring
            166: ('?', 0),  # Fragments  FIX: this can't be done in SMARTS
        }


    def get_no_change_variables(self, input_columns, objectives):
        """分子構造組込に関係のない列を抽出

        Args:
            input_columns (list): 1st argument
            objectives (list): 2nd argument
        Returns:
            None

        """
        temp_list = objectives + self.source_list
        no_change_variables = list(set(input_columns) ^ set(temp_list))
        return no_change_variables
    

    def get_smiles(self):
        """外部DB(PubChem)からのCASコードに紐づいたSMILES表記の自動取得

        Args:
            None
        Returns:
            list: self.mol_list
            dict: true_name_dict
        """
        
        if self.structure_mode=="mfp" or self.structure_mode=="maccs":
            comp_list = []
            idx_list = []
            for i in range(0,len(self.df_master)):
                if self.df_master["Source_Name"][i] in self.source_list:
                    if pd.isnull(self.df_master['CAS'][i]):
                        comp_list.append([])
                        idx_list.append(i)
                    else:
                        comp_list.append(pcp.get_compounds(self.df_master['CAS'][i],'name'))
                        idx_list.append(i)

            for i in range(0,len(comp_list)):
                if len(comp_list[i]) != 0:
                    self.mol_list.append(Chem.MolFromSmiles(comp_list[i][0].canonical_smiles))
                elif pd.isnull(self.df_master['SMILES'][idx_list[i]]) == False:
                    self.mol_list.append(Chem.MolFromSmiles(self.df_master['SMILES'][idx_list[i]]))

            true_name_dict={}
            for i, comp in enumerate(comp_list):
                try:
                    true_name_dict[self.source_list[i]] = comp[0].iupac_name
                except:
                    true_name_dict[self.source_list[i]] = "物質名がデータベースに登録されていません"

            return self.mol_list, true_name_dict
        else:
            return [], {}


    def generate_fingerprint_dataset(self, df_reg, objectives):
        """入力データから分子構造特徴表現への変換のラッパー関数
        
        Args:
            df_reg (pandas.Dataframe): 1st argument
            objectives (list): 2nd argument
        Returns:
            pandas.Dataframe: df_chem
        
        """
        if (self.structure_mode == "maccs" or self.structure_mode == "mfp") and not self.df_master.empty:
            input_columns = list(df_reg.columns)
            no_change_variables = self.get_no_change_variables(input_columns, objectives)

            if self.structure_mode=="maccs":
                df_chem = self.generate_maccs(df_reg)
                df_chem = pd.concat([df_chem.reset_index(),df_reg[objectives + no_change_variables]],axis=1).drop(['index'],axis=1)
            elif self.structure_mode=="mfp":
                df_chem = self.generate_mfp(df_reg, self.radius)
                df_chem = pd.concat([df_chem.reset_index(),df_reg[objectives + no_change_variables]],axis=1).drop(['index'],axis=1)
            return df_chem
        else:
            return df_reg
        
        
    def generate_maccs(self, df_reg):
        """SMILE表記からMaccs keyへ変換

        Args:
            df_reg (pandas.Dataframe): 1st argument
        Returns:
            pandas.Dataframe: df_maccs
            
        """
        
        maccs_list1 = [MACCSkeys.GenMACCSKeys(i).ToBitString() for i in self.mol_list]
        maccs_list2 = [list(map(int,list(i))) for i in maccs_list1]
        maccs_dict = dict(zip(self.source_list,maccs_list2))
        
        column_list = ['MACCS_' + str(i) for i in range(0,167)]
        df_maccs = pd.DataFrame(columns=column_list)
        for i, row in df_reg.iterrows():
            sum_of_maccs = np.array([0 for i in range(0,167)], dtype=np.float64)
            for j in self.source_list:
                sum_of_maccs += row[j] * np.array(maccs_dict[j])
            df_maccs = df_maccs.append(pd.DataFrame([sum_of_maccs],columns=column_list))
            df_maccs = df_maccs.drop(['MACCS_0'],axis=1)

        return df_maccs

    
    def generate_mfp(self, df_reg, radius):
        """SMILE表記からMorgan Fingerprintへ変換

        Args:
            df_reg (pandas.Dataframe): 1st argument
            radius (int): 2nd argument
        Returns:
            pandas.Dataframe: df_morgan
            
        """
        
        if not self.duplicate_check:
            bitI_key_list = []
            for i in range(len(self.mol_list)):
                bitI_morgan = {}
                AllChem.GetMorganFingerprintAsBitVect(self.mol_list[i], radius, nBits=self.bit_num, bitInfo=bitI_morgan)
                bitI_key_list.extend(list(bitI_morgan.keys()))
            bitI_unique_keys = np.unique(np.array(bitI_key_list))

            morgan_list1 = [AllChem.GetMorganFingerprintAsBitVect(i, radius, nBits=self.bit_num, bitInfo={}) for i in self.mol_list]
            morgan_list2 = [list(map(int, list(i))) for i in morgan_list1]
            morgan_list3 = []

            for j in range(len(self.mol_list)):
                temp = []
                for i in bitI_unique_keys:
                    temp.append(morgan_list2[j][i])
                morgan_list3.append(temp)
            morgan_dict = dict(zip(self.source_list, morgan_list3))

            column_list = ['bit_' + str(i) for i in bitI_unique_keys]
            np_morgan = np.zeros((len(df_reg), len(column_list)))
            for i, row in df_reg.iterrows():
                for j in self.source_list:
                    np_morgan[i] += row[j] * np.array(morgan_dict[j])
            df_morgan = pd.DataFrame(np_morgan, columns=column_list)

            return df_morgan

        else:
            error_msg = "bit数が不足しており、分子構造特徴の割り当てが重複しています。bit_num に4096以上の数字を入力してやり直してください"
            print(error_msg)
            raise Lib_ParseError(error_msg)
        
            
    def draw_chemical_structure(self, key_nums=[]):
        """分子構造特徴の可視化のラッパー関数
        
        Args:
            key_num (list): 1st argument
        Returns:
            list: name_list
        
        """

        if self.structure_mode=="maccs":
            name_list = self.draw_num_maccs(key_nums)
        elif self.structure_mode=="mfp":
            name_list = self.draw_num_mfp(key_nums)
        else:
            name_list = []
        return name_list
        
    def draw_maccs(self, key_nums):
        """指定したMACCS keyの構造パターンの可視化

        Args:
            key_nums(list): 1st argument
        Returns:
            None
            
        """

        
        def material_print(materials, k_num):
            if Local_mode == False:
                raw = '【原材料名】:' + materials
                print('=====MACCS key ' + str(k_num) + 'に対応した原材料=====')
                print(raw)

        def pattern_print(pattern):
            if Local_mode == False:
                print("【MACCS keyに対応する構造パターン（複数パターンが存在する場合には複数出力）】: {}".format(pattern))
        
        name_list = []
        for k_num in key_nums:
            try:
                smart_pattern = self.SMARTS_PATTERNS[k_num][0]
                p_name_list = []
                if k_num == 1:
                    material_print("", k_num)
                    pattern_print("None")
                    continue
                
                elif k_num == 125:
                    for i,mol in enumerate(self.mol_list):
                        ri = mol.GetRingInfo()
                        nArom = 0
                        for ring in ri.BondRings():
                            isArom = True
                            for bondIdx in ring:
                                if not mol.GetBondWithIdx(bondIdx).GetIsAromatic():
                                    isArom = False
                                    break                                
                            if isArom:
                                nArom += 1
                                if nArom > 1:
                                    p_name_list.append(str(self.source_list[i]))
                                    break
                    raw = ", ".join(p_name_list)
                    material_print(raw, k_num)
                    pattern_print("Aromatic Ring")
                    name_list.append(raw)
                    continue
                                    
                elif k_num == 166:
                    for i,mol in enumerate(self.mol_list):
                        if len(Chem.GetMolFrags(mol))>1:
                            p_name_list.append(str(self.source_list[i]))
                    raw = ", ".join(p_name_list)
                    material_print(raw, k_num)
                    pattern_print("Fragments")
                    name_list.append(raw)
                    continue

                except_key_list = [31, 48, 86, 87, 107, 149, 160]
                if len(smart_pattern.split(',')) > 1 and k_num not in except_key_list:
                    smart_list = smart_pattern.lstrip('[').rstrip(']').split(',')
                    if len(smart_list[0]) <= 3:
                        smart_list = ['[' + i + ']' for i in smart_list]
                    else:
                        smart_list = [i.lstrip('[').rstrip(']') for i in smart_list]  
                    pattern_num = len(smart_list)
                    vmol_list = [Chem.MolFromSmarts(str(smart_list[i]).lstrip('$(').rstrip(')')) for i in range(pattern_num)]
                else:
                    vmol_list = [Chem.MolFromSmarts(smart_pattern)]

                for i,mol in enumerate(self.mol_list):
                    for smart_mol in vmol_list:
                        if mol.HasSubstructMatch(smart_mol):
                            p_name_list.append(str(self.source_list[i]))
                raw = ", ".join(p_name_list)
                #material_print(raw, k_num)
                name_list.append(raw)
            
                save_folder = mfn.get_each_structure_path(self.experiment_ID, Local_mode=Local_mode)
                
                for i,vmol in enumerate(vmol_list):
                    maccs_each_filename = mfn.get_maccs_each_filename(self.experiment_ID, k_num, i, Local_mode=Local_mode)
                    try:
                        Draw.MolToFile(vmol, maccs_each_filename, size=(300,300), legend="{0:04}_{1}".format(k_num, i))
                        pattern_print(str(smart_list[i]))
                    except:
                        pattern_print("None")
                
                p_temp = list(pathlib.Path(save_folder).glob('maccs_{0:04d}_*.png'.format(k_num)))
                skimg_list = []
                for p in p_temp:
                    skimg = skimage.io.imread(p)
                    skimg_list.append(skimg)
                merge_img = skimage.util.montage(skimg_list, multichannel=True)
                merge_img_filename = mfn.get_maccs_bit_filename(self.experiment_ID, k_num, Local_mode=Local_mode)
                skimage.io.imsave(merge_img_filename, merge_img)
            except:
                continue
        return name_list


    def draw_num_maccs(self, num_list):
        """指定したMACCS keyの構造パターンの可視化

        Args:
            num_list (list): 1st argument
        Returns:
            list: name_list
            
        """
        name_list = []
        save_folder = mfn.get_each_structure_path(self.experiment_ID, Local_mode=Local_mode)
        for i in num_list:
            p_temp = list(pathlib.Path(save_folder).glob('*{0:04d}.png'.format(i)))
            if not p_temp:
                error_msg = "指定したkeyに対応する分子構造特徴が存在しません"
                print(error_msg)
                raise Lib_ParseError(error_msg)
            p_name_list = []
            for p in p_temp:
                p_stem = str(p.stem)
                p_list = p_stem.split("_")
                p_name = p_stem.replace("_{}".format(p_list[-1]), "")
                p_name_list.append(p_name)
            bit_img_filename = mfn.get_maccs_bit_filename(self.experiment_ID, i, Local_mode=Local_mode)
            shutil.copy(p_temp[0], bit_img_filename)
            
            if Local_mode:
                name_list.append(str(",\n".join(p_name_list)))
            else:
                name_list.append(str(", ".join(p_name_list)))

        return name_list


    def preview_chemical(self):
        """手法に応じた分子構造組込の変換を行い、各分子構造の画像を作成

        Args:
            None
        Returns:
            None
            
        """
        if self.structure_mode=="maccs":
            self.preview_maccs()
        elif self.structure_mode=="mfp":
            self.preview_mfp()
        else:
            pass
    
    def preview_maccs(self):
        """MACCS Keyへの変換を行い、各分子構造の画像を作成

        Args:
            None
        Returns:
            None
            
        """
        vmol_list_list = []
        vmol_list_list.append([])
        pattern_list_list = []
        pattern_list_list.append([])
        key_list_dict = {}
        except_key_list = [31, 48, 86, 87, 107, 149, 160]

        source_dict_125 = {}
        source_dict_166 = {}
        for source_name in self.source_list:
            source_dict_125[source_name] = False
            source_dict_166[source_name] = False

        for k_num in range(1, 167):
            smart_pattern = self.SMARTS_PATTERNS[k_num][0]
            raw=""
            if k_num == 1:
                vmol_list_list.append([])
                pattern_list_list.append([])
                continue

            elif k_num == 125:
                for i,mol in enumerate(self.mol_list):
                    ri = mol.GetRingInfo()
                    nArom = 0
                    for ring in ri.BondRings():
                        isArom = True
                        for bondIdx in ring:
                            if not mol.GetBondWithIdx(bondIdx).GetIsAromatic():
                                isArom = False
                                break                                
                        if isArom:
                            nArom += 1
                            if nArom > 1:
                                raw += ' ' + str(self.source_list[i])
                                source_dict_125[self.source_list[i]] = True
                                break
                vmol_list_list.append([])
                pattern_list_list.append([])
                continue

            elif k_num == 166:
                for i,mol in enumerate(self.mol_list):
                    if len(Chem.GetMolFrags(mol))>1:
                        raw += ' ' + str(self.source_list[i])
                        source_dict_166[self.source_list[i]] = True
                vmol_list_list.append([])
                pattern_list_list.append([])
                continue

            if len(smart_pattern.split(',')) > 1 and k_num not in except_key_list:
                smart_list = smart_pattern.lstrip('[').rstrip(']').split(',')
                if len(smart_list[0]) <= 3:
                    smart_list = ['[' + i + ']' for i in smart_list]
                else:
                    smart_list = [i.lstrip('[').rstrip(']') for i in smart_list]
                pattern_num = len(smart_list)
                vmol_list = [Chem.MolFromSmarts(str(smart_list[i]).lstrip('$(').rstrip(')')) for i in range(pattern_num)]
                pattern_list = smart_list
            else:
                vmol_list = [Chem.MolFromSmarts(smart_pattern)]
                pattern_list = [smart_pattern]
            vmol_list_list.append(vmol_list)
            pattern_list_list.append(pattern_list)
    
        for s_idx, source_name in enumerate(self.source_list):
            key_list = []
            for key, vmol_list in enumerate(vmol_list_list):
                for smart_mol in vmol_list:
                    mol = self.mol_list[s_idx]
                    if mol.HasSubstructMatch(smart_mol):
                        key_list.append(key)
                        break
            if source_dict_125[source_name] == True:
                key_list.append(125)
            if source_dict_166[source_name] == True:
                key_list.append(166)
            key_list_dict[source_name]=key_list
        
        save_folder = mfn.get_each_structure_path(self.experiment_ID, Local_mode=Local_mode)
        error_nums=[1, 125, 166]
        for source_name in self.source_list:
            for k_num in key_list_dict[source_name]:
                maccs_each_filename = os.path.join(save_folder, "{0}_{1:04}.png".format(source_name, k_num))            
                if k_num not in error_nums:
                    img_list = []
                    for i in range(len(vmol_list_list[k_num])):
                        img = Draw.MolToImage(vmol_list_list[k_num][i], legend="{0:04}_{1:02} ({2})".format(k_num, i, pattern_list_list[k_num][i]), size=(300, 300))
                        img_list.append(img)

                    def convert_1d_to_2d(l, cols):
                        return [l[i:i + cols] for i in range(0, len(l), cols)]

                    col_num = math.ceil(math.sqrt(len(img_list)))
                    img_2d_list = convert_1d_to_2d(img_list, col_num)

                    get_concat_tile_blank(img_2d_list).save(maccs_each_filename)
                else:
                    ini = Image.new("RGB", (100, 100), (255, 255, 255))
                    ini.save(maccs_each_filename)

    
    def preview_mfp(self):
        """Morgan Fingerprintの特徴一覧を出力

        Args:
            None
        Returns:
            bool: duplicate_check
            
        """
        if self.structure_mode == "mfp":
            each_folder = mfn.get_each_structure_path(self.experiment_ID, Local_mode=Local_mode)
            list_folder = mfn.get_list_structure_path(self.experiment_ID, Local_mode=Local_mode)
            mfn.clear_folder(each_folder)
            mfn.clear_folder(list_folder)
            
            bitI_key_list = []
            bitI_morgan_list = []
            for i in range(len(self.mol_list)):
                bitI_morgan = {}
                AllChem.GetMorganFingerprintAsBitVect(self.mol_list[i], self.radius, nBits=self.bit_num, bitInfo=bitI_morgan)
                bitI_key_list.extend(list(bitI_morgan.keys()))
                bitI_morgan_list.append(bitI_morgan)

                for tup in bitI_morgan.values():
                    check_radius = []
                    for t in tup:
                        check_radius.append(t[1])
                    checker = all([x==check_radius[0] for x in check_radius])
                    if not checker:
                        self.duplicate_check = True

            bitI_unique_keys = np.unique(np.array(bitI_key_list))

            source_filename = mfn.get_source_filename(self.experiment_ID, Local_mode=Local_mode)
            f = open(source_filename, 'w')
            f.write("index" + "," + "source name" + '\n')
            for i in bitI_unique_keys:
                source_name_list = []
                for j, source_name in enumerate(self.source_list):
                    if i in bitI_morgan_list[j]:
                        source_name_list.append(source_name)
                raw = str(i) + "," + " / ".join(source_name_list)
                f.write(raw + '\n')
            f.close()

            if not self.duplicate_check:
                for i in range(len(self.mol_list)):
                    chem_name = self.source_list[i]
                    print('#' + chem_name)
                    bit_list = list(bitI_morgan_list[i].keys())
                    skimg_list = []

                    for bit_n in bit_list:
                        mfp_each_filename = mfn.get_mfp_each_filename(self.experiment_ID, chem_name, bit_n, Local_mode=Local_mode)
                        img = Draw.DrawMorganBit(self.mol_list[i], bit_n, bitI_morgan_list[i], legend="{0}_{1:04d}".format(chem_name, bit_n))
                        img.save(mfp_each_filename)
                        skimg = skimage.io.imread(mfp_each_filename)
                        skimg_list.append(skimg)
                    merge_img = skimage.util.montage(skimg_list, multichannel=True)
                    merge_img_filename = mfn.get_mfp_structure_list_filename(self.experiment_ID, chem_name, Local_mode=Local_mode)      
                    skimage.io.imsave(merge_img_filename, merge_img)
                    print('出力先: {}'.format(merge_img_filename))

            return self.duplicate_check


    def draw_num_mfp(self, num_list):
        """指定したMorgan Fingerprintの構造パターンの可視化

        Args:
            num_list(list): 1st argument
        Returns:
            list: name_list
            
        """
        if self.structure_mode=="mfp":
            
            bitI_morgan_list = []
            for i in range(len(self.mol_list)):
                bitI_morgan = {}
                AllChem.GetMorganFingerprintAsBitVect(self.mol_list[i], self.radius, nBits=self.bit_num, bitInfo=bitI_morgan)
                bitI_morgan_list.append(bitI_morgan)

            name_list = []
            save_folder = mfn.get_each_structure_path(self.experiment_ID, Local_mode=Local_mode)
            for i in num_list:
                p_temp = list(pathlib.Path(save_folder).glob('*{0:04d}.png'.format(i)))
                if not p_temp:
                    error_msg = "指定したbitに対応する分子構造特徴が存在しません"
                    print(error_msg)
                    raise Lib_ParseError(error_msg)
                p_name_list = []
                for p in p_temp:
                    p_stem = str(p.stem)
                    p_list = p_stem.split("_")
                    p_name = p_stem.replace("_{}".format(p_list[-1]), "")
                    p_name_list.append(p_name)
                chem_loc = self.source_list.index(p_name_list[0])
                img = Draw.DrawMorganBit(self.mol_list[chem_loc], i, bitI_morgan_list[chem_loc], legend="bit={0}".format(i))
                bit_img_filename = mfn.get_mfp_bit_filename(self.experiment_ID, i, Local_mode=Local_mode)
                img.save(bit_img_filename)

                if Local_mode:
                    name_list.append(str(",\n".join(p_name_list)))
                else:
                    name_list.append(str(", ".join(p_name_list)))

            return name_list

        else:
            error_msg = "bit数が不足しており、分子構造特徴の割り当てが重複しています。bit_num に4096以上の数字を入力してやり直してください"
            print(error_msg)
            raise Lib_ParseError(error_msg)

        
    def draw_topnum(self, top_num, objectives):
        """Shap値上位の分子構造特徴の可視化

        Args:
            top_num(int): 1st argument
            objectives (list): 2nd argument
        Returns:
            list: source_name_list
            
        """
        
        if (self.structure_mode == "mfp" or self.structure_mode == "maccs") and len(self.source_list) != 0:
            output_folder = mfn.get_bit_structure_path(self.experiment_ID, Local_mode=Local_mode)
            p_list_structure = list(pathlib.Path(output_folder).glob('shap_result_*.png'))
            for img_name in p_list_structure:
                if os.path.exists(img_name):
                    os.remove(img_name)

            source_name_list = []
            for target in objectives:
                shapresult_filename = mfn.get_shapresult_target_filename(self.experiment_ID, target, Local_mode=Local_mode)
                json_open = open(shapresult_filename, 'r')
                json_load = json.load(json_open)
                json_data = json_load["explanations"]["kernel_shap"]["label0"]["global_shap_values"]
                json_sorted = sorted(json_data.items(), key=lambda x:x[1], reverse=True)

                # shap値上位の分子構造特徴の番号を抽出
                key_nums = []
                rank_nums = []
                for i, (k, v) in enumerate(json_sorted):
                    key_num_split = k.split("_")
                    if len(key_num_split) > 1 and (key_num_split[0] == "bit" or key_num_split[0] == "MACCS"):
                        try:
                            key_num = int(key_num_split[1])
                            key_nums.append(key_num)
                            rank_nums.append(i+1)
                        except:
                            continue
                    if len(key_nums) == top_num:
                        break

                name_list = self.draw_chemical_structure(key_nums)
                source_name_list.append(name_list)

                skimg_list = []
                for i, (rank, key) in enumerate(zip(rank_nums, key_nums)):
                    if self.structure_mode == "mfp":
                        print("目的変数: {}".format(target))
                        print("第{0}位 bit {1}".format(rank, key))
                        print("【対応する原材料名】: {}".format(name_list[i]))
                        img_filename = mfn.get_mfp_bit_filename(self.experiment_ID, key, Local_mode=Local_mode)
                    elif self.structure_mode == "maccs":
                        print("目的変数: {}".format(target))
                        print("第{0}位 key {1}".format(rank, key))
                        print("【対応する原材料名】: {}".format(name_list[i]))
                        img_filename = mfn.get_maccs_bit_filename(self.experiment_ID, key, Local_mode=Local_mode)
                    if os.path.exists(img_filename):
                        new_img_filename = mfn.get_shaptop_target_filename(self.experiment_ID, rank, target, Local_mode=Local_mode)
                        os.rename(img_filename, new_img_filename)
                        skimg = skimage.io.imread(new_img_filename)
                        skimg_list.append(skimg)

                if self.structure_mode == "mfp":
                    skimg_col = 5 if len(skimg_list) > 5 else len(skimg_list)
                    merge_img = skimage.util.montage(skimg_list, multichannel=True, grid_shape=(math.ceil(len(skimg_list)/skimg_col), skimg_col))
                    merge_img_filename = mfn.get_shaprank_target_filename(self.experiment_ID, target, Local_mode=Local_mode)
                    skimage.io.imsave(merge_img_filename, merge_img)
                    print('出力先: {}\n'.format(merge_img_filename))

            return source_name_list
        
        else:
            print("分子構造組み込み分析は選択されていません")
            return []