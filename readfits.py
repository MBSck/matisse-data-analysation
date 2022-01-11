#!/usr/bin/env python3

__author__ = "Marten Scheuck"

class ReadoutFits:
    def __init__(self, fits_file: Path):
        self.file = fits_file
        hdu = fits.open(f)

    def get_data(self, header):
        return hdu[header].data["data"][:6], \
                hdu[header].data[header + "err"][:6]

    @property
    def vis2(self):
        return get_data('oi_vis2')

    @property
    def t3phi(self):
        return get_data('oi_t3')[0][:4], get_data('oi_t3')[0][:4]


