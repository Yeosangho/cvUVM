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

#ifndef _UTILS_H_
#define _UTILS_H_

struct PaletteEntry
{
    unsigned char b, g, r, a;
};

#define WRITE_PIX( ptr, clr )       \
    (((uchar*)(ptr))[0] = (clr).b,  \
     ((uchar*)(ptr))[1] = (clr).g,  \
     ((uchar*)(ptr))[2] = (clr).r)

#define  descale(x,n)  (((x) + (1 << ((n)-1))) >> (n))
#define  saturate(x)   (uchar)(((x) & ~255) == 0 ? (x) : ~((x)>>31))

void icvCvt_BGR2Gray_8u_C3C1R( const uchar* bgr, int bgr_step,
                               uchar* gray, int gray_step,
                               CvSize size, int swap_rb=0 );


CV_INLINE bool  isBigEndian( void )
{
    return (((const int*)"\0\x1\x2\x3\x4\x5\x6\x7")[0] & 255) != 0;
}

#endif/*_UTILS_H_*/
