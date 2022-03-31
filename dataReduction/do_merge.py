import os

from shutil import copyfile
from astropy.io import fits

"""Slight rewrite of Jozsef's code and folder wide application"""

def oifits_patchwork(incoherent_file_path: str, coherent_file: str,
                     outfile_path: str,
                     oi_types_list=[['vis2','visamp','visphi','t3','flux']],
                     headerval=[]) -> None:
    if os.path.exists(incoherent_file):
        copyfile(incoherent_file, outfile_path)
    else:
        raise RuntimeError('ERROR (oifits_patchwork): File not found: '+incoherent_file)

    outhdul  = fits.open(outfile_path, mode='update')

    n_oi_types_list = len(oi_types_list)
    for i in range(n_oi_types_list):
        oi_types = oi_types_list[i]
        inhdul = fits.open(incoherent_file, mode='readonly')

        for oi_type in oi_types:
            if oi_type == 'vis2':
                outhdul['OI_VIS2'].data = inhdul['OI_VIS2'].data
            if oi_type == 't3':
                outhdul['OI_T3'].data = inhdul['OI_T3'].data
            if oi_type == 'visamp':
                try:
                    outhdul[0].header['HIERARCH ESO PRO CAL NAME']          =          inhdul[0].header['HIERARCH ESO PRO CAL NAME']
                    outhdul[0].header['HIERARCH ESO PRO CAL RA']            =            inhdul[0].header['HIERARCH ESO PRO CAL RA']
                    outhdul[0].header['HIERARCH ESO PRO CAL DEC']           =           inhdul[0].header['HIERARCH ESO PRO CAL DEC']
                    outhdul[0].header['HIERARCH ESO PRO CAL AIRM']          =          inhdul[0].header['HIERARCH ESO PRO CAL AIRM']
                    outhdul[0].header['HIERARCH ESO PRO CAL FWHM']          =          inhdul[0].header['HIERARCH ESO PRO CAL FWHM']
                    outhdul[0].header['HIERARCH ESO PRO CAL TAU0']          =          inhdul[0].header['HIERARCH ESO PRO CAL TAU0']
                    outhdul[0].header['HIERARCH ESO PRO CAL TPL START']     =     inhdul[0].header['HIERARCH ESO PRO CAL TPL START']
                    outhdul[0].header['HIERARCH ESO PRO CAL DB NAME']       =       inhdul[0].header['HIERARCH ESO PRO CAL DB NAME']
                    outhdul[0].header['HIERARCH ESO PRO CAL DB DBNAME']     =     inhdul[0].header['HIERARCH ESO PRO CAL DB DBNAME']
                    outhdul[0].header['HIERARCH ESO PRO CAL DB RA']         =         inhdul[0].header['HIERARCH ESO PRO CAL DB RA']
                    outhdul[0].header['HIERARCH ESO PRO CAL DB DEC']        =        inhdul[0].header['HIERARCH ESO PRO CAL DB DEC']
                    outhdul[0].header['HIERARCH ESO PRO CAL DB DIAM']       =       inhdul[0].header['HIERARCH ESO PRO CAL DB DIAM']
                    outhdul[0].header['HIERARCH ESO PRO CAL DB ERRDIAM']    =    inhdul[0].header['HIERARCH ESO PRO CAL DB ERRDIAM']
                    # outhdul[0].header['HIERARCH ESO PRO CAL DB SEPARATION'] = inhdul[0].header['HIERARCH ESO PRO CAL DB SEPARATION']
                except KeyError as e:
                    print(e)

            if oi_type == 'flux':
                try:
                    outhdul['OI_FLUX'].data = inhdul['OI_FLUX'].data
                except KeyError as e:
                    pass

            infile2 = coherent_file
            inhdul2 = fits.open(infile2, mode='readonly')

            outhdul['OI_VIS'].header['AMPTYP'] = inhdul2['OI_VIS'].header['AMPTYP']
            outhdul['OI_VIS'].data = inhdul2['OI_VIS'].data

            #look up visphi
            if 'visphi' in oi_types:
                visphi = inhdul2['OI_VIS'].data['VISPHI']
                visphierr = inhdul2['OI_VIS'].data['VISPHIERR']
                #match station indices
                sta_index_visamp = outhdul['OI_VIS'].data['STA_INDEX']
                sta_index_visamp = [ list(item) for item in sta_index_visamp ]
                sta_index_visphi = inhdul2['OI_VIS'].data['STA_INDEX']
                sta_index_visphi = [ list(item) for item in sta_index_visphi ]
                for k in range(len(sta_index_visamp)):
                    for l in range(len(sta_index_visphi)):
                        if ((sta_index_visamp[k] == sta_index_visphi[l]) \
                            or (sta_index_visamp[k][::-1] == sta_index_visphi[l] )):
                            outhdul['OI_VIS'].data['VISPHI'][k] = inhdul2['OI_VIS'].data['VISPHI'][l]
                            outhdul['OI_VIS'].data['VISPHIERR'][k] = inhdul2['OI_VIS'].data['VISPHIERR'][l]

    for dic in headerval:
        del outhdul[0].header[dic['key']]
        outhdul[0].header[dic['key']] = dic['value']


    outhdul.flush()  # changes are written back to original.fits
    outhdul.close()
    inhdul.close()
    inhdul2.close()

def do_single_merge(

def merge_files(product_folder: str, both: bool = False,
                lband: bool = False) -> None:
    if both:
    for i in ["lband", "nband"]:
        incoherent_folders = glob(os.path.join(product_folder, f"incoherent/{i}/calib", "*.rb"))
        coherent_folders = glob(os.path.join(product_folder, f"coherent_folders/{i}/calib", "*.rb"))
        print(incoherent_folders, coherent_folders)
        outfile_dir = os.path.join(base_folder, "combined", i)

        if not os.path.exists(outfile_dir):
            os.makedirs(outfile_dir)

        outfile_path = os.path.join(outfile_dir, os.path.basename(coherent_file))
        oifits_patchwork(incoherent_file, coherent_file, outfile_path)
    else:
        band = "lband" if lband else "nband"


if __name__ == "__main__":
    base_folder = "/data/beegfs/astro-storage/groups/matisse/scheuck/data/GTO/hd142666/PRODUCTS/20190514/"
    merge_files(base_folder, both=True)

