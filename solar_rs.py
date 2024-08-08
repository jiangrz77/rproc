import numpy as np
from util import Z2Symbol_dict

Z_arr = np.array([
    31, 32, 33, 34, 35, 36, 37, 38, 39, 40,  
    41, 42, 44, 45, 46, 47, 48, 49, 50, 
    51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 
    62, 63, 64, 65, 66, 67, 68, 69, 70, 
    71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 
    81, 82, 83, 90])

logeps_sol_31_s = 2.93
logeps_sol_32_s = 3.37
logeps_sol_33_s = 2.06
logeps_sol_34_s = 2.91
logeps_sol_35_s = 1.96
logeps_sol_36_s = 2.88
logeps_sol_37_s = 2.08
logeps_sol_38_s = 2.84
logeps_sol_39_s = 2.07
logeps_sol_40_s = 2.46
logeps_sol_41_s = 1.22
logeps_sol_42_s = 1.61
logeps_sol_44_s = 1.30
logeps_sol_45_s = 0.16
logeps_sol_46_s = 1.30
logeps_sol_47_s = 0.52
logeps_sol_48_s = 1.45
logeps_sol_49_s = 0.33
logeps_sol_50_s = 1.91
logeps_sol_51_s = 0.40
logeps_sol_52_s = 1.49
logeps_sol_53_s = 0.06
logeps_sol_54_s = 1.51
logeps_sol_55_s = 0.28
logeps_sol_56_s = 2.11
logeps_sol_57_s = 1.07
logeps_sol_58_s = 1.51
logeps_sol_59_s = 0.47
logeps_sol_60_s = 1.22
logeps_sol_62_s = 0.44
logeps_sol_63_s = -0.81
logeps_sol_64_s = 0.29
logeps_sol_65_s = -0.83
logeps_sol_66_s = 0.30
logeps_sol_67_s = -0.66
logeps_sol_68_s = 0.19
logeps_sol_69_s = -0.77
logeps_sol_70_s = 0.55
logeps_sol_71_s = -0.61
logeps_sol_72_s = 0.48
logeps_sol_73_s = -0.47
logeps_sol_74_s = 0.43
logeps_sol_75_s = -0.54
logeps_sol_76_s = 0.35
logeps_sol_77_s = -0.61
logeps_sol_78_s = 0.51
logeps_sol_79_s = -0.44
logeps_sol_80_s = 0.92
logeps_sol_81_s = 0.65
logeps_sol_82_s = 1.95
logeps_sol_83_s = -0.02
logeps_sol_90_s = np.nan

logeps_sol_31_r = 2.51
logeps_sol_32_r = 3.13
logeps_sol_33_r = 1.92
logeps_sol_34_r = 3.13
logeps_sol_35_r = 2.41
logeps_sol_36_r = 3.02
logeps_sol_37_r = 2.06
logeps_sol_38_r = 1.78
logeps_sol_39_r = 1.52
logeps_sol_40_r = 1.80
logeps_sol_41_r = 0.94
logeps_sol_42_r = 1.34
logeps_sol_44_r = 1.53
logeps_sol_45_r = 1.02
logeps_sol_46_r = 1.37
logeps_sol_47_r = 1.10
logeps_sol_48_r = 1.33
logeps_sol_49_r = 0.53
logeps_sol_50_r = 1.52
logeps_sol_51_r = 0.88
logeps_sol_52_r = 2.04
logeps_sol_53_r = 1.54
logeps_sol_54_r = 2.16
logeps_sol_55_r = 1.01
logeps_sol_56_r = 1.23
logeps_sol_57_r = 0.47
logeps_sol_58_r = 0.77
logeps_sol_59_r = 0.41
logeps_sol_60_r = 1.04
logeps_sol_62_r = 0.75
logeps_sol_63_r = 0.48
logeps_sol_64_r = 0.98
logeps_sol_65_r = 0.28
logeps_sol_66_r = 1.04
logeps_sol_67_r = 0.44
logeps_sol_68_r = 0.83
logeps_sol_69_r = 0.06
logeps_sol_70_r = 0.68
logeps_sol_71_r = -0.01
logeps_sol_72_r = 0.32
logeps_sol_73_r = -0.47
logeps_sol_74_r = 0.25
logeps_sol_75_r = 0.20
logeps_sol_76_r = 1.29
logeps_sol_77_r = 1.33
logeps_sol_78_r = 1.58
logeps_sol_79_r = 0.77
logeps_sol_80_r = 0.82
logeps_sol_81_r = 0.15
logeps_sol_82_r = 1.25
logeps_sol_83_r = 0.54
logeps_sol_90_r = 0.15
logeps_sol_92_r = -0.11

logeps_sols = np.array([
    logeps_sol_31_s,logeps_sol_32_s,logeps_sol_33_s,logeps_sol_34_s,logeps_sol_35_s,logeps_sol_36_s,logeps_sol_37_s,logeps_sol_38_s,logeps_sol_39_s,logeps_sol_40_s, 
    logeps_sol_41_s,logeps_sol_42_s,logeps_sol_44_s,logeps_sol_45_s,logeps_sol_46_s,logeps_sol_47_s,logeps_sol_48_s,logeps_sol_49_s,logeps_sol_50_s,
    logeps_sol_51_s,logeps_sol_52_s,logeps_sol_53_s,logeps_sol_54_s,logeps_sol_55_s,logeps_sol_56_s,logeps_sol_57_s,logeps_sol_58_s,logeps_sol_59_s,logeps_sol_60_s,
    logeps_sol_62_s,logeps_sol_63_s,logeps_sol_64_s,logeps_sol_65_s,logeps_sol_66_s,logeps_sol_67_s,logeps_sol_68_s,logeps_sol_69_s,logeps_sol_70_s,
    logeps_sol_71_s,logeps_sol_72_s,logeps_sol_73_s,logeps_sol_74_s,logeps_sol_75_s,logeps_sol_76_s,logeps_sol_77_s,logeps_sol_78_s,logeps_sol_79_s,logeps_sol_80_s,
    logeps_sol_81_s,logeps_sol_82_s,logeps_sol_83_s,logeps_sol_90_s])

logeps_solr = np.array([
    logeps_sol_31_r,logeps_sol_32_r,logeps_sol_33_r,logeps_sol_34_r,logeps_sol_35_r,logeps_sol_36_r,logeps_sol_37_r,logeps_sol_38_r,logeps_sol_39_r,logeps_sol_40_r, 
    logeps_sol_41_r,logeps_sol_42_r,logeps_sol_44_r,logeps_sol_45_r,logeps_sol_46_r,logeps_sol_47_r,logeps_sol_48_r,logeps_sol_49_r,logeps_sol_50_r,
    logeps_sol_51_r,logeps_sol_52_r,logeps_sol_53_r,logeps_sol_54_r,logeps_sol_55_r,logeps_sol_56_r,logeps_sol_57_r,logeps_sol_58_r,logeps_sol_59_r,logeps_sol_60_r,
    logeps_sol_62_r,logeps_sol_63_r,logeps_sol_64_r,logeps_sol_65_r,logeps_sol_66_r,logeps_sol_67_r,logeps_sol_68_r,logeps_sol_69_r,logeps_sol_70_r,
    logeps_sol_71_r,logeps_sol_72_r,logeps_sol_73_r,logeps_sol_74_r,logeps_sol_75_r,logeps_sol_76_r,logeps_sol_77_r,logeps_sol_78_r,logeps_sol_79_r,logeps_sol_80_r,
    logeps_sol_81_r,logeps_sol_82_r,logeps_sol_83_r,logeps_sol_90_r])

logeps_sols_dict = {
    Z2Symbol_dict[_z]: 
    logeps_sols[_idx] for _idx, _z in enumerate(Z_arr)
}
logeps_solr_dict = {
    Z2Symbol_dict[_z]: 
    logeps_solr[_idx] for _idx, _z in enumerate(Z_arr)
}