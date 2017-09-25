from __future__ import division
from scipy.special import eval_legendre, lpmv
import numpy as np
import LFPy
import time
import matplotlib.pyplot as plt

def calc_vn(n):
    r_const = (r34 ** (2*n + 1) - 1) / ((n + 1) / n * r34 ** (2*n + 1) + 1)
    v = (n / (n + 1) * sigma34 - r_const) / (sigma34 + r_const)
    return v

def calc_yn(n):
    vn = calc_vn(n)
    # r_const1 = (n / (n + 1) * r23 ** n - vn * r32 ** (n + 1)) / (r23 ** n + vn * r32 ** (n + 1))
    r_const = (n / (n + 1) * r23 ** (2*n + 1) - vn) / (r23 ** (2*n + 1) + vn)
    y = (n / (n + 1) * sigma23 - r_const) / (sigma23 + r_const)
    return y

def calc_zn(n):
    yn = calc_yn(n)
    # z = (r12 ** n - (n + 1) / n * yn * r21 ** (n + 1)) / (r12 ** n + yn * r21 ** (n + 1))
    z = (r12 ** (2*n+1) - (n + 1) / n * yn) / (r12 ** (2*n+1) + yn)
    return z

def calc_c1n(n):
    zn = calc_zn(n)
    c = ((n + 1) / n * sigma12 + zn) / (sigma12 - zn) * rz1**(n+1)
    return c

def calc_c2n(n):
    yn = calc_yn(n)
    c1 = calc_c1n(n)
    c2 = (c1 + rz1**(n+1)) * r12 ** (n + 1) / (r12 ** (2 * n + 1) + yn)
    return c2

def calc_d2n(n, c2):
    yn = calc_yn(n)
    d2 = yn * c2
    return d2

def calc_c3n(n):
    vn = calc_vn(n)
    c2 = calc_c2n(n)
    d2 = calc_d2n(n, c2)
    c3 = (c2 + d2) * r23 ** (n + 1) / (r23 ** (2*n + 1) + vn)
    return c3

def calc_d3n(n, c3):
    vn = calc_vn(n)
    d3 = vn * c3
    return d3

def calc_c4n(n):
    c3 = calc_c3n(n)
    d3 = calc_d3n(n, c3)
    c4 = (n + 1) / n * r34 ** (n + 1) * (c3 + d3) / ((n + 1) / n * r34 ** (2*n + 1) + 1)
    return c4

def calc_d4n(n, c4):
    d4 = n / (n + 1) * c4
    return d4

def calc_csf_term1(n, r):
    # print('n: ', n)
    yn = calc_yn(n)
    c1 = calc_c1n(n)
    # term1 = (c1 + rz1 ** (n + 1)) / ((r1 / r) ** n + (r2 ** 2 / (r1 * r)) ** n * r21 * yn)
    term1 = (c1 + rz1**(n+1))*r12*((r1*r)/(r2**2))**n/(r12**(2*n+1) + yn)
    # print('term1: ', term1, 'term1_2', term1_2, 'diff', term1 - term1_2)
    return term1

def calc_csf_term2(n, r):
    yn = calc_yn(n)
    c1 = calc_c1n(n)
    term2 = yn*(c1 + rz1 ** (n + 1)) / (r/r2*((r1 * r) / r2**2) ** n  + (r / r1) ** (n+1)*yn)
    # term2 = (r / r1) ** (n+1)
    return term2


if __name__ == '__main__':

    brain = False
    csf = True
    skull = False
    scalp = False

    radii = [79000, 80000., 85000., 90000.]
    sigmas = [0.3, 1.5, 0.015, 0.3]

    r1 = radii[0]; r2 = radii[1]; r3 = radii[2]; r4 = radii[3]
    r12 = r1 / r2; r21 = r2 / r1
    r23 = r2 / r3; r32 = r3 / r2
    r34 = r3 / r4; r43 = r4 / r3

    sigma1 = sigmas[0]; sigma2 = sigmas[1]; sigma3 = sigmas[2]; sigma4 = sigmas[3]
    sigma12 = sigma1 / sigma2; sigma21 = sigma2 / sigma1
    sigma23 = sigma2 / sigma3; sigma32 = sigma3 / sigma2
    sigma34 = sigma3 / sigma4; sigma43 = sigma4 / sigma3



    if brain:
        rz_list = np.array([77000., 77100., 77200., 77400., 77500., 77600., 77700., 77800., 77900., 77990.])
        rz1_list = rz_list/r1
        r = 78000.
        r1r_list = [1.001, 1.0001, 1.00001, 1.000001, 1.00000001]
        r1_list = r*np.array(r1r_list)

        N_list = []

        for i in range(len(r1_list)):
            r1 = r1_list[i]
            n_list = []
            for j in range(len(rz_list)):
                rz = rz_list[j]
                rz1 = rz/r
                const = 1.
                consts = 0.
                n = 0
                # while const > 1e-30:
                while const > 1e-10*consts:
                    n += 1
                    c1n = calc_c1n(n)
                    const = n*(c1n * (r / r1) ** n + (rz / r) ** (n + 1))
                    consts += const
                print ('n', n, 'rz', rz, 'consts', consts, 'consts[-1]', const)
                n_list.append(n)
            N_list.append(n_list)
        fig_title = 'brain'
        x = rz1_list[:len(n_list)]
        xlab = 'rz/r1'
        r_list = r1_list
        lbl = 'r1/r = '
        lb_list = r1r_list

    if csf:
        # make rr1 rz1 csf fig
        # rr1_list = [1.+ 1e-2, 1.+ 1e-3, 1.+ 1e-4, 1.+ 1e-5, 1.+ 1e-6, 1.+ 1e-7, 1.+ 1e-8, 1.+ 1e-9, 1.+ 1e-10]
        # r_list = r1*np.array(rr1_list)
        r_list = np.array([79000.001])  #, 79000.1])
        # make rr2 rz1 csf fig
        # rr2_list = [1.- 1e-3, 1.- 1e-4, 1.- 1e-5, 1.- 1e-6, 1.- 1e-7, 1.- 1e-8, 1.- 1e-9, 1.- 1e-10]
        # r_list = r2*np.array(rr2_list)
        rz_list = [78999.]
        rz1_list = np.array(rz_list)/r1
        N_list = []
        num_iterations = 1e6
        for j in range(len(r_list)):
            r = r_list[j]
            print('rr1: ', r/r1)
            n_list = []
            for i in range(len(rz_list)):
                rz = rz_list[i]
                rz1 = rz/r1
                const = 1.
                consts = 0.
                const_list = np.zeros(num_iterations)
                start_time = time.clock()
                n = 0
                # while const > 1e-10*consts:
                while const > 2./99.*1e-6*consts:
                    n += 1
                    term1 = calc_csf_term1(n,r)
                    term2 = calc_csf_term2(n,r)
                    const = n*(term1 + term2)
                    consts += const
                    const_list[n-1] = consts
                total_time = time.clock() - start_time
                print ('n', n, 'rz', rz, 'consts', consts, 'consts[-1]', const)
                n_list.append(n)
            N_list.append(n_list)
        # np.save('./convergence_plots/consts_79001_78900.npy', const_list)
    #     fig_title = 'csf_5'
    #     x = rz_list[:len(n_list)]
    #     # x = rz1_list[:len(N_list)]
    #     xlab = 'rz/r1'
    #     # x_ticklbls = [str(r1 - i) for i in rz_list]
    #     lbl = 'r = '
    #     lb_list = r_list
    #
    # if skull:
    #     # make rr2 rz1 skull fig
    #     # rr2_list = [1.+ 1e-2, 1.+ 1e-3, 1.+ 1e-4, 1.+ 1e-5, 1.+ 1e-6, 1.+ 1e-7, 1.+ 1e-8, 1.+ 1e-9, 1.+ 1e-10]
    #     # r_list = r2*np.array(rr2_list)
    #     # make rr3 rz1 skull fig
    #     rr3_list = [1.- 1e-3, 1.- 1e-4, 1.- 1e-5, 1.- 1e-6, 1.- 1e-7, 1.- 1e-8, 1.- 1e-9, 1.- 1e-10]
    #     r_list = r3*np.array(rr3_list)
    #     rz_list = [77000., 77300., 77700., 78000., 78100., 78200., 78400., 78500., 78600., 78700., 78800., 78900., 78990., 78999., 78999.999]
    #     rz1_list = np.array(rz_list)/r1
    #     N_list = []
    #     for j in range(len(r_list)):
    #         r = r_list[j]
    #         n_list = []
    #         for i in range(len(rz_list)):
    #             rz = rz_list[i]
    #             rz1 = rz/r1
    #             const = 1.
    #             consts = 0.
    #             n = 0
    #             while const > 1e-10*consts:
    #             # while n < 10000:
    #                 n += 1
    #                 c3n = calc_c3n(n)
    #                 d3n = calc_d3n(n, c3n)
    #                 const = n*(c3n * (r / r3) ** n + d3n * (r3 / r) ** (n + 1))
    #                 consts += const
    #             print ('n', n, 'rz', rz, 'consts', consts, 'consts[-1]', const)
    #             n_list.append(n)
    #         N_list.append(n_list)
    #     fig_title = 'skull_r3'
    #     x = rz1_list[:len(n_list)]
    #     lbl = 'r/r3 = '
    #     lb_list = rr3_list
    #     # x = rz1_list[:len(N_list)]
    #     xlab = 'rz/r1'
    #
    # if scalp:
    #     # make rr2 rz1 skull fig
    #     # rr3_list = [1.+ 1e-2, 1.+ 1e-3, 1.+ 1e-4, 1.+ 1e-5, 1.+ 1e-6, 1.+ 1e-7, 1.+ 1e-8, 1.+ 1e-9, 1.+ 1e-10]
    #     # r_list = r3*np.array(rr3_list)
    #     # make rr3 rz1 skull fig
    #     rr4_list = [1.- 1e-3, 1.- 1e-4, 1.- 1e-5, 1.- 1e-6, 1.- 1e-7, 1.- 1e-8, 1.- 1e-9, 1.- 1e-10]
    #     r_list = r4*np.array(rr4_list)
    #     rz_list = [77000., 77300., 77700., 78000., 78100., 78200., 78400., 78500., 78600., 78700., 78800., 78900., 78990., 78999., 78999.999]
    #     rz1_list = np.array(rz_list)/r1
    #     N_list = []
    #     for j in range(len(r_list)):
    #         r = r_list[j]
    #         n_list = []
    #         for i in range(len(rz_list)):
    #             rz = rz_list[i]
    #             rz1 = rz/r1
    #             const = 1.
    #             consts = 0.
    #             n = 0
    #             while np.abs(const) > 1e-10*consts:
    #             # while n < 100:
    #                 n += 1
    #                 print('n:', n)
    #                 c4n = calc_c4n(n)
    #                 print('c4n:', c4n)
    #                 d4n = calc_d4n(n, c4n)
    #                 print('d4n:', d4n)
    #                 const = n*(c4n * (r / r4) ** n + d4n * (r4 / r) ** (n + 1))
    #                 consts += const
    #                 print('const:', const)
    #             print ('n', n, 'rz', rz, 'consts', consts, 'consts[-1]', const)
    #             n_list.append(n)
    #         N_list.append(n_list)
    #     fig_title = 'scalp_r4'
    #     x = rz1_list[:len(n_list)]
    #     lbl = 'r/r4 = '
    #     lb_list = rr4_list
    #     # x = rz1_list[:len(N_list)]
    #     xlab = 'rz/r1'
    #
    # colorbrewer = {'lightblue': '#a6cee3', 'blue': '#1f78b4', 'lightgreen': '#b2df8a',
    #                'green': '#33a02c', 'pink': '#fb9a99', 'red': '#e31a1c',
    #                'lightorange': '#fdbf6f', 'orange': '#ff7f00',
    #                'lightpurple': '#cab2d6', 'purple': '#6a3d9a',
    #                'yellow': '#ffff33', 'brown': '#b15928'}
    # clrs = colorbrewer.values()[:len(r_list)]
    # # clrs = colorbrewer.values()[:len(rr2_list)]
    # plt.close('all')
    # fig = plt.figure()
    # for ii in range(len(N_list)):
    #     plt.plot(x, N_list[ii], label= lbl + str(lb_list[ii]), color=clrs[ii], linewidth=2.)
    #     # plt.plot(x, N_list[ii], label= 'r/r1 = ' + str(rr1_list[ii]), color=clrs[ii], linewidth=2.)
    #     # plt.plot(x, N_list[ii], label= 'r/r2 = ' + str(rr2_list[ii]), color=clrs[ii], linewidth=2.)
    # plt.plot()
    # plt.xlabel(xlab)
    # plt.ylabel('n')
    # fig.axes[0].spines['top'].set_visible(False)
    # fig.axes[0].spines['right'].set_visible(False)
    # fig.axes[0].get_xaxis().tick_bottom()
    # fig.axes[0].get_yaxis().tick_left()
    # # if csf:
    # #     x_ticklbls = fig.axes[0].get_x
    # #     fig.axes[0].set_xticklabels(x_ticklbls)
    # plt.title(fig_title)
    # plt.legend(loc=2)
    # file_name = './convergence_plots/convergence_' + fig_title + '.png'
    # plt.savefig(file_name)
    #
    #
    #
    #     # r = 82000.
    #     # rz = 78000.
    #     # rz1 = rz/r1
    #     # n_list = [1, 10, 100, 1000, 10000, 20000]
    #     # for n in n_list:
    #     #     c2n = calc_c2n(n)
    #     #     d2n = calc_d2n(n, c2n)
    #     #     csf_const1 = n*(c2n * (r / r2) ** n + d2n * (r2 / r) ** (n + 1))
    #     #     term1 = calc_csf_term1(n,r)
    #     #     term2 = calc_csf_term2(n,r)
    #     #     csf_const2 = n*(term1 + term2)
    #     #     print('1:', csf_const1, '2:', csf_const2, 'diff', np.abs(csf_const1 - csf_const2))
    #
    #
    # #
    # #             n = np.arange(1, 100)
    # #             c1n = self._calc_c1n(n)
    # #             consts = n*(c1n * (r / self.r1) ** n + (self.rz / r) ** (n + 1))
    # #             consts = np.insert(consts, 0, 0) # since the legendre function starts with P0
    # #             leg_consts = np.polynomial.legendre.Legendre(consts)
    # #             pot_sum = leg_consts(np.cos(theta))
    # #             print('consts[-1]', consts[-1])
    # #             n2 = np.arange(1, 1000)
    # #             c1n2 = self._calc_c1n(n2)
    # #             consts2 = n2*(c1n2 * (r / self.r1) ** n2 + (self.rz / r) ** (n2 + 1))
    # #             consts2 = np.insert(consts2, 0, 0) # since the legendre function starts with P0
    # #             leg_consts2 = np.polynomial.legendre.Legendre(consts2)
    # #             pot_sum2 = leg_consts2(np.cos(theta))
    # #             print('potsum1', pot_sum, 'potsum2', pot_sum2, 'diff', pot_sum - pot_sum2)
    # #             return pot_sum
