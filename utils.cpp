/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of Intel Corporation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "utils.hpp"

#define  SCALE  14
#define  cR  (int)(0.299*(1 << SCALE) + 0.5)
#define  cG  (int)(0.587*(1 << SCALE) + 0.5)
#define  cB  ((1 << SCALE) - cR - cG)

void icvCvt_BGR2Gray_8u_C3C1R( const uchar* rgb, int rgb_step,
                               uchar* gray, int gray_step,
                               CvSize size, int _swap_rb )
{
    int i;
    int swap_rb = _swap_rb ? 2 : 0;
    for( ; size.height--; gray += gray_step )
    {
        for( i = 0; i < size.width; i++, rgb += 3 )
        {
            int t = descale( rgb[swap_rb]*cB + rgb[1]*cG + rgb[swap_rb^2]*cR, SCALE );
            gray[i] = (uchar)t;
        }

        rgb += rgb_step - size.width*3;
    }
}

