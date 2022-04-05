import os

from glob import glob
from astropy.io import fits

def avg_oifits(infile_list,outfile_path,headerval=[]):
    if os.path.exists(infile_list[0]):
        copyfile(infile_list[0], outfile_path)
    else:
        print('ERROR (avg_oifits): File not found: '+infile_list[0])
        return
    outhdul  = fits.open(outfile_path, mode='update')

    inhdul_lst = []
    visamp_lst = []
    visamperr_lst = []
    visphi_lst = []
    visphierr_lst = []
    visamp_ucoord_lst = []
    visamp_vcoord_lst = []
    visamp_sta_index_lst = []
    vis2_lst = []
    vis2err_lst = []
    vis2_ucoord_lst = []
    vis2_vcoord_lst = []
    vis2_sta_index_lst = []
    t3phi_lst = []
    t3phierr_lst = []
    t3phi_u1coord_lst = []
    t3phi_v1coord_lst = []
    t3phi_u2coord_lst = []
    t3phi_v2coord_lst = []
    t3phi_sta_index_lst = []
    flux_lst = []
    fluxerr_lst = []

    for infile in infile_list:
        #read OI_VIS
        if os.path.exists(infile):
            inhdul_lst.append(fits.open(infile, mode='readonly'))

            # print(infile)
            #read OI_VIS
            visamp = inhdul_lst[-1]['OI_VIS'].data['VISAMP']
            visamperr = inhdul_lst[-1]['OI_VIS'].data['VISAMPERR']
            visphi = inhdul_lst[-1]['OI_VIS'].data['VISPHI']
            visphierr = inhdul_lst[-1]['OI_VIS'].data['VISPHIERR']
            sta_index = inhdul_lst[-1]['OI_VIS'].data['STA_INDEX']
            ucoord = inhdul_lst[-1]['OI_VIS'].data['UCOORD']
            vcoord = inhdul_lst[-1]['OI_VIS'].data['VCOORD']
            for i in range(len(visamp)):
                if np.all(visamp[i] == 0.0):
                    visamp[i] = visamp[i]*np.nan
                    visamperr[i] = visamperr[i]*np.nan
                if np.all(visphi[i] == 0.0):
                    visphi[i] = visphi[i]*np.nan
                    visphierr[i] = visphierr[i]*np.nan
                visamp_lst.append(visamp[i])
                visamperr_lst.append(visamperr[i])
                visphi_lst.append(visphi[i])
                visphierr_lst.append(visphierr[i])
                visamp_sta_index_lst.append(sta_index[i])
                visamp_ucoord_lst.append(ucoord[i])
                visamp_vcoord_lst.append(vcoord[i])

            #read OI_VIS2
            vis2 = inhdul_lst[-1]['OI_VIS2'].data['VIS2DATA']
            vis2err = inhdul_lst[-1]['OI_VIS2'].data['VIS2ERR']
            sta_index = inhdul_lst[-1]['OI_VIS2'].data['STA_INDEX']
            ucoord = inhdul_lst[-1]['OI_VIS2'].data['UCOORD']
            vcoord = inhdul_lst[-1]['OI_VIS2'].data['VCOORD']
            for i in range(len(vis2)):
                if np.all(vis2[i] == 0.0):
                    vis2[i] = vis2[i]*np.nan
                    vis2err[i] = vis2err[i]*np.nan
                vis2_lst.append(vis2[i])
                vis2err_lst.append(vis2err[i])
                vis2_sta_index_lst.append(sta_index[i])
                vis2_ucoord_lst.append(ucoord[i])
                vis2_vcoord_lst.append(vcoord[i])

            #read OI_T3
            t3phi = inhdul_lst[-1]['OI_T3'].data['T3PHI']
            t3phierr = inhdul_lst[-1]['OI_T3'].data['T3PHIERR']
            sta_index = inhdul_lst[-1]['OI_T3'].data['STA_INDEX']
            u1coord = inhdul_lst[-1]['OI_T3'].data['U1COORD']
            v1coord = inhdul_lst[-1]['OI_T3'].data['V1COORD']
            u2coord = inhdul_lst[-1]['OI_T3'].data['U2COORD']
            v2coord = inhdul_lst[-1]['OI_T3'].data['V2COORD']
            for i in range(len(t3phi)):
                if np.all(t3phi[i] == 0.0):
                    t3phi[i] = t3phi[i]*np.nan
                    t3phierr[i] = t3phierr[i]*np.nan
                t3phi_lst.append(t3phi[i])
                t3phierr_lst.append(t3phierr[i])
                t3phi_sta_index_lst.append(sta_index[i])
                t3phi_u1coord_lst.append(u1coord[i])
                t3phi_v1coord_lst.append(v1coord[i])
                t3phi_u2coord_lst.append(u2coord[i])
                t3phi_v2coord_lst.append(v2coord[i])

            is_flux = True
            #read OI_FLUX
            try:
                fluxdata = inhdul_lst[-1]['OI_FLUX'].data['FLUXDATA']
                fluxerr  = inhdul_lst[-1]['OI_FLUX'].data['FLUXERR']
                for spectrum,errspectrum in zip(fluxdata,fluxerr):
                    if np.all(spectrum == 0.0):
                        flux_lst.append(spectrum*np.nan)
                        fluxerr_lst.append(errspectrum*np.nan)
                    else:
                        flux_lst.append(spectrum)
                        fluxerr_lst.append(errspectrum)
            except KeyError as e:
                # print(e)
                is_flux = False
                flux_lst.append(np.nan*visamp)
                fluxerr_lst.append(np.nan*visamp)
        else:
            print('WARNING (avg_oifits): File not found: '+infile)

    if not inhdul_lst:
        outhdul.close()
        os.remove(outfile_path)
        print('ERROR (avg_oifits): No files to average.')
        return

    #average fluxes:
    if is_flux == True:
        flux_arr = np.array(flux_lst)
        fluxerr_arr = np.array(fluxerr_lst)
        # for ii in range(len(flux_arr)):
        #     print(flux_arr[ii])
        #avg_flux = np.nanmean(flux_arr,axis=0)
        avg_flux = robust.mean(flux_arr,axis=0)
        #avg_flux = np.nanmedian(flux_arr,axis=0)
        # print(avg_flux)
        # print(len(flux_arr))
        if len(flux_arr) > 3:
            # combine two error sources: standard deviation over the different BCDs, and average error (calculated by the pipeline)
            avg_fluxerr = np.sqrt(np.nanstd(flux_arr,axis=0)**2.0 + np.nanmean(fluxerr_arr,axis=0)**2.0)
        else:
            avg_fluxerr = np.nanmean(fluxerr_arr,axis=0) #WARNING: it may be not the best method for error calculation
        outhdul['OI_FLUX'].data = outhdul['OI_FLUX'].data[0:1]
        outhdul['OI_FLUX'].data['FLUXDATA'] = avg_flux
        outhdul['OI_FLUX'].data['FLUXERR'] = avg_fluxerr

    # collect unique station indices from OI_VIS
    sta_index_unique_lst = []
    ucoord_unique_lst = []
    vcoord_unique_lst = []
    sta_index = inhdul_lst[0]['OI_VIS'].data['STA_INDEX']
    sta_index= [ list(item) for item in sta_index ]
    ucoord = inhdul_lst[0]['OI_VIS'].data['UCOORD']
    vcoord = inhdul_lst[0]['OI_VIS'].data['VCOORD']
    sta_index_unique_lst.append(sta_index[0])
    ucoord_unique_lst.append(ucoord[0])
    vcoord_unique_lst.append(vcoord[0])
    for i in range(1,len(sta_index)):
        if not ( (sta_index[i] in sta_index_unique_lst) \
        or (sta_index[i][::-1] in sta_index_unique_lst )):
            sta_index_unique_lst.append(sta_index[i])
            ucoord_unique_lst.append(ucoord[i])
            vcoord_unique_lst.append(vcoord[i])

    #average VISAMP and VISPHI
    n_sta_index = len(sta_index_unique_lst)
    outhdul['OI_VIS'].data = outhdul['OI_VIS'].data[0:n_sta_index]
    for k in range(len(sta_index_unique_lst)):
        #collect and average matching visamp data
        visamp_lst_sta = []
        visamperr_lst_sta = []
        visphi_lst_sta = []
        visphierr_lst_sta = []
        for i in range(len(visamp_sta_index_lst)):
            if (((sta_index_unique_lst[k][0] == visamp_sta_index_lst[i][0]) and (sta_index_unique_lst[k][1] == visamp_sta_index_lst[i][1])) \
            or ((sta_index_unique_lst[k][0] == visamp_sta_index_lst[i][1]) and (sta_index_unique_lst[k][1] == visamp_sta_index_lst[i][0]))):
                visamp_lst_sta.append(visamp_lst[i])
                visamperr_lst_sta.append(visamperr_lst[i])
                visphi_lst_sta.append(visphi_lst[i])
                visphierr_lst_sta.append(visphierr_lst[i])
        visamp_arr = np.array(visamp_lst_sta)
        visamperr_arr = np.array(visamperr_lst_sta)
        visphi_arr = np.array(visphi_lst_sta)
        visphierr_arr = np.array(visphierr_lst_sta)
        avg_visamp = np.nanmean(visamp_arr,axis=0)
        avg_visphi = np.arctan2(np.nanmean(np.sin(visphi_arr*np.pi/180.0),axis=0),np.nanmean(np.cos(visphi_arr*np.pi/180.0),axis=0))*180.0/np.pi
        # print(len(visamp_arr))
        if len(visamp_arr) > 3:
            avg_visamperr = np.sqrt(np.nanstd(visamp_arr,axis=0)**2.0 + np.nanmean(visamperr_arr,axis=0)**2.0)
            avg_visphierr = np.sqrt(np.nanstd(visphi_arr,axis=0)**2.0 + np.nanmean(visphierr_arr,axis=0)**2.0)
            # combine two error sources: standard deviation over the different BCDs, and average error (calculated by the pipeline)
        else:
            avg_visamperr = np.nanmean(visamperr_arr,axis=0) #WARNING: it may be not the best method for error calculation
            avg_visphierr = np.nanmean(visphierr_arr,axis=0)
        # print(avg_visamp)
        # print(outhdul['OI_VIS'].data['VISAMP'][k])
        outhdul['OI_VIS'].data['VISAMP'][k] = avg_visamp
        outhdul['OI_VIS'].data['VISAMPERR'][k] = avg_visamperr
        outhdul['OI_VIS'].data['VISPHI'][k] = avg_visphi
        outhdul['OI_VIS'].data['VISPHIERR'][k] = avg_visphierr
        outhdul['OI_VIS'].data['STA_INDEX'][k] = sta_index_unique_lst[k]
        outhdul['OI_VIS'].data['UCOORD'][k] = ucoord_unique_lst[k]
        outhdul['OI_VIS'].data['VCOORD'][k] = vcoord_unique_lst[k]

    # collect unique station indices from OI_VIS2
    sta_index_unique_lst = []
    ucoord_unique_lst = []
    vcoord_unique_lst = []
    sta_index = inhdul_lst[0]['OI_VIS2'].data['STA_INDEX']
    sta_index= [ list(item) for item in sta_index ]
    ucoord = inhdul_lst[0]['OI_VIS2'].data['UCOORD']
    vcoord = inhdul_lst[0]['OI_VIS2'].data['VCOORD']
    sta_index_unique_lst.append(sta_index[0])
    ucoord_unique_lst.append(ucoord[0])
    vcoord_unique_lst.append(vcoord[0])
    for i in range(1,len(sta_index)):
        if not ( (sta_index[i] in sta_index_unique_lst) \
        or (sta_index[i][::-1] in sta_index_unique_lst )):
            sta_index_unique_lst.append(sta_index[i])
            ucoord_unique_lst.append(ucoord[i])
            vcoord_unique_lst.append(vcoord[i])

    #average VIS2
    n_sta_index = len(sta_index_unique_lst)
    outhdul['OI_VIS2'].data = outhdul['OI_VIS2'].data[0:n_sta_index]
    for k in range(len(sta_index_unique_lst)):
        #collect and average matching vis2 data
        vis2_lst_sta = []
        vis2err_lst_sta = []
        for i in range(len(vis2_sta_index_lst)):
            if (((sta_index_unique_lst[k][0] == vis2_sta_index_lst[i][0]) and (sta_index_unique_lst[k][1] == vis2_sta_index_lst[i][1])) \
            or ((sta_index_unique_lst[k][0] == vis2_sta_index_lst[i][1]) and (sta_index_unique_lst[k][1] == vis2_sta_index_lst[i][0]))):
                vis2_lst_sta.append(vis2_lst[i])
                vis2err_lst_sta.append(vis2err_lst[i])
        vis2_arr = np.array(vis2_lst_sta)
        vis2err_arr = np.array(vis2err_lst_sta)
        avg_vis2 = np.nanmean(vis2_arr,axis=0)
        if len(vis2_arr) > 3:
            # combine two error sources: standard deviation over the different BCDs, and average error (calculated by the pipeline)
            avg_vis2err = np.sqrt(np.nanstd(vis2_arr,axis=0)**2.0 + np.nanmean(vis2err_arr,axis=0)**2.0)
        else:
            avg_vis2err = np.nanmean(vis2err_arr,axis=0) #WARNING: it may be not the best method for error calculation
        outhdul['OI_VIS2'].data['VIS2DATA'][k] = avg_vis2
        outhdul['OI_VIS2'].data['VIS2ERR'][k] = avg_vis2err
        outhdul['OI_VIS2'].data['STA_INDEX'][k] = sta_index_unique_lst[k]
        outhdul['OI_VIS2'].data['UCOORD'][k] = ucoord_unique_lst[k]
        outhdul['OI_VIS2'].data['VCOORD'][k] = vcoord_unique_lst[k]

    # collect unique station indices from OI_T3
    sta_index_unique_lst = []
    sta_index_unique_lst_sorted = []
    u1coord_unique_lst = []
    v1coord_unique_lst = []
    u2coord_unique_lst = []
    v2coord_unique_lst = []
    sta_index = inhdul_lst[0]['OI_T3'].data['STA_INDEX']
    sta_index= [ list(item) for item in sta_index ]
    u1coord = inhdul_lst[0]['OI_T3'].data['U1COORD']
    v1coord = inhdul_lst[0]['OI_T3'].data['V1COORD']
    u2coord = inhdul_lst[0]['OI_T3'].data['U2COORD']
    v2coord = inhdul_lst[0]['OI_T3'].data['V2COORD']
    sta_index_unique_lst.append(sta_index[0])
    sta_index_unique_lst_sorted.append(sorted(sta_index[0]))
    # print(sorted(sta_index[0]))
    u1coord_unique_lst.append(u1coord[0])
    v1coord_unique_lst.append(v1coord[0])
    u2coord_unique_lst.append(u2coord[0])
    v2coord_unique_lst.append(v2coord[0])
    for i in range(1,len(sta_index)):
        if not ( (sorted(sta_index[i]) in sta_index_unique_lst_sorted) ):
            sta_index_unique_lst.append(sta_index[i])
            sta_index_unique_lst_sorted.append(sorted(sta_index[i]))
            u1coord_unique_lst.append(u1coord[i])
            v1coord_unique_lst.append(v1coord[i])
            u2coord_unique_lst.append(u2coord[i])
            v2coord_unique_lst.append(v2coord[i])

    #average T3PHI
    n_sta_index = len(sta_index_unique_lst)
    outhdul['OI_T3'].data = outhdul['OI_T3'].data[0:n_sta_index]
    for k in range(len(sta_index_unique_lst)):
        #collect and average matching vis2 data
        t3phi_lst_sta = []
        t3phierr_lst_sta = []
        # print('k',k,sta_index_unique_lst_sorted[k])
        for i in range(len(t3phi_sta_index_lst)):
            # print('i',i,sorted(t3phi_sta_index_lst[i]))
            if sta_index_unique_lst_sorted[k] == sorted(t3phi_sta_index_lst[i]):
                t3phi_lst_sta.append(t3phi_lst[i])
                t3phierr_lst_sta.append(t3phierr_lst[i])
        t3phi_arr = np.array(t3phi_lst_sta)
        t3phierr_arr = np.array(t3phierr_lst_sta)
        #avg_t3phi = np.nanmean(t3phi_arr,axis=0)
        avg_t3phi = np.arctan2(np.nanmean(np.sin(t3phi_arr*np.pi/180.0),axis=0),np.nanmean(np.cos(t3phi_arr*np.pi/180.0),axis=0))*180.0/np.pi
        if len(t3phi_arr) > 3:
            # combine two error sources: standard deviation over the different BCDs, and average error (calculated by the pipeline)
            avg_t3phierr = np.sqrt(np.nanstd(t3phi_arr,axis=0)**2.0 + np.nanmean(t3phierr_arr,axis=0)**2.0)
        else:
            avg_t3phierr = np.nanmean(t3phierr_arr,axis=0) #WARNING: it may be not the best method for error calculation
        outhdul['OI_T3'].data['T3PHI'][k] = avg_t3phi
        outhdul['OI_T3'].data['T3PHIERR'][k] = avg_t3phierr
        outhdul['OI_T3'].data['STA_INDEX'][k] = sta_index_unique_lst[k]
        outhdul['OI_T3'].data['U1COORD'][k] = u1coord_unique_lst[k]
        outhdul['OI_T3'].data['V1COORD'][k] = v1coord_unique_lst[k]
        outhdul['OI_T3'].data['U2COORD'][k] = u2coord_unique_lst[k]
        outhdul['OI_T3'].data['V2COORD'][k] = v2coord_unique_lst[k]

    for dic in headerval:
        outhdul[0].header[dic['key']] = dic['value']

    outhdul.flush()  # changes are written back to original.fits
    outhdul.close()
    for inhdul in inhdul_lst:
        inhdul.close()

def average_files(product_folder: str, both: bool = False,
                lband: bool = False) -> None:
        # TODO: Add lband functionality and difference between chopped 1-4 and
        # non chopped exposures 5-6. Do not average those together
        for i in ["nband"]:
            folders = glob(os.path.join(product_folder, f"combined/{i}", "*.rb"))
            outfile_dir = os.path.join(product_folder, "combined/averaged", i)

            for j, k in enumerate(folders):
                print(f"Averaging all files in {os.path.basename(k)}")
                print("------------------------------------------------------------")
                outfile_dir = os.path.join(outfile_dir, os.path.basename(k))

                if not os.path.exists(outfile_dir):
                    os.makedirs(outfile_dir)

                fits_files = glob(os.path.join(k, "*.fits"))
                fits_files.sort(key=lambda x: x[-8:])

                outfile_path = os.path.join(outfile_dir, os.path.basename(m))
                avg_oifits(fits_files, outfile_path)
                print(f"Done averaging all files in {os.path.basename(k)}")
                print("------------------------------------------------------------")

if __name__ == "__main__":
    folder = ""
    average_files(folder)


