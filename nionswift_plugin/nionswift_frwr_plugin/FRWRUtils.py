import numpy as np
import os, sys, glob
from pathlib import Path
import h5py
import time
import site

import nion.data.xdata_1_0 as xd

def dict_get_value(d,k):
    result = None
    if k in d.keys():
        result = d[k]
    else:
        for v in d.values():
            if isinstance(v,dict):
                result = dict_get_value(v,k)
                if (result is not None):
                    break
    return result

class DefocusSetup:
    def __init__(self, data0, metadata0):
        #sets default values
        #reset when loading new data
        
        ''' Copies image data/metadata to class attributes. Image data is rearranged such that first index in array is the slice number. '''
        self.data0 = data0
        self.data = data0[:,:,:].copy()
        first_axis = np.nonzero(self.data.shape == np.min(self.data.shape))[0][0]
        dims = np.arange(0,self.data.ndim)
        self.dims_adj = dims[dims != first_axis] #old axis arrangement is saved, since the metadata concerning axes is still in old format
        self.data = np.transpose(self.data,(first_axis,*self.dims_adj))
        self.metadata = metadata0.copy()
        self.data_fft = None #FFT of the defocus stack
        self.slice_data = None #slice_data and slice_data_fft are stacks of the images actually displayed in nionswift, including annotations etc.
        self.slice_data_fft = None
        
        ''' Internal parameters that can't be adjusted from the GUI '''
        self.stepsFoc = self.data.shape[0]
        self.centralSlice = self.stepsFoc//2
        self.currentSlice = 0
        self.height = len(self.data[0])
        self.width = len(self.data[0,0])
        self.scaleX = 0.0
        self.scaleY = 0.0
        self.wvl = 0.001 #nm
        
        ''' Adjustable parameters that change the Thon ring calculation and are saved in the hdf5 output file '''
        ''' Defaults set here are overriden by the stack metadata if present '''
        self.DeltaFoc = 2.0 #nm
        self.offsetFoc = 0.0 #nm
        self.apertRadius = 15.0 #mrad
        self.apertOffsetX = 0.0 #mrad
        self.apertOffsetY = 0.0 #mrad
        self.astigMag = 0.0 #nm
        self.astigAngle = 0.0 #deg
        self.Cs = 1.2 #mm
        self.highVoltage = 200.0 #kV
        self.CTFpoints = 12
        self.CTFrings = 2
        self.focusExp = 2.0
        
        ''' Array of defocus values for each slice in the stack '''
        self.defocusVal = np.empty(int(self.stepsFoc))
        
        ''' Flags changing behaviour of defocus calculation and thon ring display '''
        self.RingFlag = True #Sets whether whole rings are drawn designating the CTF zero crossings or only separate points of the rings (amount set by CTFpoints)
        self.nonLinearFlag = True #Sets whether defocus step is linear or not
        
        ''' Adjustable parameters that don't affect anything (just passed to the saved hdf5 file) '''
        self.Delta = 10.0 #nm
        self.alpha = 0.1 #mrad
        self.apertEdge = 0.1 #fraction of ObjectiveAperture
        self.expTime = 1.0 #s

        ''' Internal parameters for manual defocus setting. Defocus offset and defocus step are calculated from defocus manually set for two slices. Setting defocus for further slices will compare set defocus with the previously adjusted slice '''
        self.fixSlice = -1 #Slice index of slice 1 used for calculation
        self.fixDef = 0 #Defocus of slice 1
        self.fixSlicePrev = 0 #Previous Slice 1 (used in case same slice is adjusted multiple times)
        self.fixDefPrev = 0
        
        ''' Arrays containing annotation masks that are applied to slice_data/slice_data_fft in update_slice_data() '''
        self.slice_annot = np.full((self.stepsFoc,self.height,self.width),False)
        self.slice_fft_annot = np.full((self.stepsFoc,self.height,self.width),False)
        self.apert_ring = np.full((self.height,self.width),False)
        self.apert_mask = np.full((self.height,self.width),False)
        
        ''' Dictionary relating variable names to their respective key in the stack metadata '''
        self.metadata_keys = {"DeltaFoc" : "DeltaFoc", "offsetFoc" : "offsetFoc", "expTime" : "expTime", "Delta" : "Delta", "alpha" : "alpha", "apertRadius" : "ObjectiveAperture", "apertEdge" : "ObjectiveApertureEdge", "apertOffsetX" : "ObjectiveApertureOffsetX", "apertOffsetY" : "ObjectiveApertureOffsetY", "astigMag" : "AstigMag", "astigAngle" : "AstigAngle", "Cs" : "Cs", "highVoltage" : "Voltage", "focusExp" : "focusScale"}
        
        
        ''' Startup functions '''
        self.calc_fft()
        self.slice_data = self.data.copy()
        self.slice_data_fft = self.data_fft.copy()
        
        self.load_defaults()
        self.load_params_from_metadata()
        self.draw_apert()
        for i in range(0,int(self.stepsFoc)):
            self.update_slice_data(i)
    

    def load_defaults(self):
        path = Path([x for x in site.getsitepackages() if "site-packages" in x][0] + "/nionswift_plugin/nionswift_frwr_plugin/defaults.dat")
        dftfile = open(path)
        for line in dftfile:
            key = line.split()
            attr = getattr(self,key[0])
            setattr(self,key[0],type(attr)(key[1]))
    
    def load_params_from_metadata(self):
        #loads values for the Defocus parameters from given metadata using metadata_keys to correspond variable <-> key in metadata
        
        for par in self.metadata_keys.keys():
            val = dict_get_value(self.metadata, self.metadata_keys[par])
            if (val is not None):
                setattr(self, par, val)
             
        if (self.highVoltage > 3000):
            self.highVoltage *= 1e-3
        if (self.apertRadius > 0):
            self.apertEdge = self.apertEdge/self.apertRadius
        
        self.scaleY = self.metadata["dimensional_calibrations"][self.dims_adj[0]]["scale"]
        self.scaleX = self.metadata["dimensional_calibrations"][self.dims_adj[1]]["scale"]
        self.get_wavelength()
        
    def calc_fft(self):
        #calculates FFT of data stack and writes it to data_fft
        self.data_fft = np.empty(self.data.shape, dtype=complex)
        for i in range(0,len(self.data)):
            self.data_fft[i] = xd.fft(self.data[i])
            
    def update_slice_data(self,slice_index):
        #applies annotations by giving the respective values a large value
        #in case of the aperture mask, i.e. the area blocked by the aperture a low value is chosen. 0 can't be used as this messes up the contrast in nionswift
        
        self.slice_data[slice_index] = self.data[slice_index].copy()
        self.slice_data_fft[slice_index] = self.data_fft[slice_index].copy()
        self.slice_data[slice_index][self.slice_annot[slice_index]] = 1000000        
        self.slice_data_fft[slice_index][self.slice_fft_annot[slice_index]] = 1000000+0j
        
        self.slice_data_fft[slice_index][self.apert_ring] = 1000000+0j
        self.slice_data_fft[slice_index][self.apert_mask] = 0.00001+0j
        #print(self.apert_mask)
            
        
        
    def draw_apert(self):
        #creates a ring corresponding to the aperture size as well as marks all the pixels blocked by the aperture (aperture_mask)
        dkx = 1.0/(self.width*self.scaleX)
        dky = 1.0/(self.height*self.scaleY)

        self.get_wavelength()

        kap = np.sin(1e-3*self.apertRadius)/self.wvl;
        t = self.height/2-kap/dky
        l = self.width/2-kap/dkx
        b = self.height/2+kap/dky
        r = self.width/2+kap/dkx
        
        self.apert_ring = self.draw_ellipse(t,l,b,r)
        self.apert_mask = self.draw_ellipse_fill(t,l,b,r,True)

            
    def createThonRings(self,slice_index):
        #calculates coordinates of points on the thon rings
        #either the amount set in CTFpoints or just two if RingFlag is set. An ellipse is then drawn using the two known points
        
        ringCount = self.CTFrings
        phiCount = 1
        astAngle = 0
        if (not self.RingFlag):
            phiCount = self.CTFpoints
            astAngle = self.astigAngle
        rad = 10
        cx = self.width/2
        cy = self.height/2
        dkx = 1.0/(self.width*self.scaleX)
        dky = 1.0/(self.height*self.scaleY)
        defoc = 0
        lambda3Cs = 0
        
        self.get_wavelength()
        
        kap = np.sin(1e-3*self.apertRadius)/self.wvl
        rad = kap/(30*dky)
        if (self.nonLinearFlag):
            defoc = self.offsetFoc+np.sign(slice_index-self.centralSlice)*(abs(slice_index-self.centralSlice)**self.focusExp)*self.DeltaFoc
        else:
            defoc = self.offsetFoc+(slice_index-self.centralSlice)*self.DeltaFoc
            
        self.defocusVal[slice_index] = defoc
            
        if (self.Cs != 0):
            lambda3Cs = 2.0/(self.wvl**3*self.Cs*1.0e6)
            
        annotation_coords= np.zeros((ringCount,phiCount,5))
        
        for iRing in range(0,ringCount):
            kRingg = np.array([])
            kRinglg = np.array([])
            kRingll = np.array([])
            if (self.RingFlag):
                phi = np.array([0,np.pi/2])
            else:
                phi = np.linspace(0,2*np.pi,phiCount,endpoint = False)
            defLoc = defoc+self.astigMag*np.cos(2*(phi+astAngle*np.pi/180))
            p=defLoc*self.wvl*lambda3Cs
            pg = p[p>0]
            phig = phi[p>0]
            pl = p[p<=0]
            phil = phi[p<=0]
            philg = np.array([])
            phill = np.array([])
            
            if(len(pg)>0):
                q = (iRing+1)*lambda3Cs
                kRingg = np.sqrt(-0.5*pg+np.sqrt(0.25*pg**2+q))/dkx
            if(len(pl)>0):
                q2min = -0.5*pl
                nMin = np.ceil(-q2min**2/lambda3Cs)
                
                plg = pl[nMin <= -(1+iRing)]
                philg = phil[nMin <= -(1+iRing)]
                pll = pl[nMin > -(1+iRing)]
                phill = phil[nMin > -(1+iRing)]
                if (len(plg) > 0):
                    q = -(iRing+1)*lambda3Cs
                    kRinglg = np.sqrt(-0.5*plg-np.sqrt(0.25*plg**2+q))/dkx
                if (len(pll) > 0):
                    q = (nMin[nMin > -(1+iRing)]+iRing)*lambda3Cs
                    kRingll = np.sqrt(-0.5*pll+np.sqrt(0.25*pll**2+q))/dkx
             
            kRing = np.concatenate((kRingg,kRinglg,kRingll))
            phicon = np.concatenate((phig,philg,phill))
            if (self.RingFlag):
                kx = np.zeros((1,1))
                ky = np.zeros((1,1))
                rada = np.array([[kRing[0]]])
                radb = np.array([[kRing[1]]])
                rotphi = np.array([[self.astigAngle*np.pi/180]])
            else:
                kx = kRing * np.cos(phicon)
                ky = kRing * np.sin(phicon)*dkx/dky
                kx = np.reshape(kx,(len(kx),1))
                ky = np.reshape(ky,(len(ky),1)) 
                rada = np.full((phiCount,1),rad)
                radb = np.full((phiCount,1),rad)
                rotphi = np.full((phiCount,1),0)
            annotation_coords[iRing] = np.concatenate((cx+kx,cy+ky,rada,radb,rotphi),1)
        
        self.update_slice_data(slice_index)
        return annotation_coords
        
    def update_defocus(self,newdef):
        #updates defocus offset and step by calculating them from manually set defocus on two slices
        #the specific calculation varies if two different slices have been changed after each other, or one slice was changed multiple times etc.
        #the statements below just decide which calculation to perform
        
        self.defocusVal[self.currentSlice] = newdef
        if (self.fixSlice < 0):
            if(self.currentSlice != self.centralSlice):
                if (self.nonLinearFlag):
                    frameStep = np.sign(self.currentSlice - self.centralSlice)*abs(self.currentSlice - self.centralSlice)**self.focusExp
                else:
                    frameStep = self.currentSlice-self.centralSlice
                self.DeltaFoc = (newdef-self.offsetFoc)/frameStep
            else:
                self.offsetFoc = newdef
            
            self.fixSlice = self.currentSlice
            self.fixDef = newdef
            self.fixSlicePrev = 0
            self.fixDefPrev = self.offsetFoc
        else:
            if (self.currentSlice != self.fixSlice):
                if (self.nonLinearFlag):
                    frameStep = np.sign(self.currentSlice - self.centralSlice)*abs(self.currentSlice - self.centralSlice)**self.focusExp
                    frameFix = np.sign(self.fixSlice-self.centralSlice)*abs(self.fixSlice-self.centralSlice)**self.focusExp
                else:
                    frameStep = self.currentSlice-self.centralSlice
                    frameFix = self.fixSlice-self.centralSlice
                
                self.DeltaFoc = (newdef-self.fixDef)/(frameStep-frameFix)
                self.offsetFoc = newdef-frameStep*self.DeltaFoc
                self.fixSlicePrev = self.fixSlice
                self.fixDefPrev = self.fixDef
                self.fixSlice = self.currentSlice
            else:
                if(self.currentSlice != self.fixSlicePrev):
                    if(self.nonLinearFlag):
                        frameStep = np.sign(self.currentSlice - self.centralSlice)*abs(self.currentSlice - self.centralSlice)**self.focusExp
                        frameFix = np.sign(self.fixSlicePrev - self.centralSlice)*abs(self.fixSlicePrev - self.centralSlice)**self.focusExp
                    else:
                        frameStep = self.currentSlice - self.centralSlice
                        frameFix = self.fixSlicePrev - self.centralSlice
                    
                    self.DeltaFoc = (newdef-self.fixDefPrev)/(frameStep-frameFix)
                    self.offsetFoc = newdef-frameStep*self.DeltaFoc
                else:
                    if (self.nonLinearFlag):
                        frameStep = np.sign(self.currentSlice - self.centralSlice)*abs(self.currentSlice - self.centralSlice)**self.focusExp
                    else:
                        frameStep = self.currentSlice-self.centralSlice
                    self.DeltaFoc = (newdef-self.offsetFoc)/frameStep
            self.fixDef = newdef
                
            
    def get_wavelength(self):
        #voltage in kV
        erest = 510.99906 #keV electron rest mass
        hc = 1.23984 #eV*um 
        self.wvl = hc/np.sqrt(self.highVoltage*(2*erest+(self.highVoltage)))
        
    def draw_ellipse(self,xtop,ytop,xbot,ybot):
        #draws an ellipse using the passed coordinates of the corners of the rectangle containing the ellipse
        
        a = (xbot-xtop)/2
        b = (ybot-ytop)/2
        wid = 0.05*max(a,b)
        wid = max(wid,3)
        wid = min(wid,5)
        a2 = a-wid
        b2 = b*(a2/a)
        x0 = xtop + a
        y0 = ytop + b
        
        
        x = np.arange(0,self.width)
        y = np.arange(0,self.height)[:,np.newaxis]
        
        ellipse1 = (((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1).astype(int) 
        ellipse2 = (((x-x0)/a2)**2 + ((y-y0)/b2)**2 <= 1).astype(int)
        ellipse = (ellipse1-ellipse2).astype(bool)
        return ellipse
    
    def draw_ellipse_fill(self,xtop,ytop,xbot,ybot,val):
        #same thing as draw_ellipse but fills it instead of making a ring
        
        a = (xbot-xtop)/2
        b = (ybot-ytop)/2
        x0 = xtop + a
        y0 = ytop + b
        
        x = np.arange(0,self.width)
        y = np.arange(0,self.height)[:,np.newaxis]
        
        ellipse = (((x-x0)/a)**2 + ((y-y0)/b)**2 <= 1)
        
        if (val):
            ellipse = ~ellipse
            
        return ellipse
        
    def save(self,dims,file_path):
        #saves the parameters in an hdf5 file
        
        if (file_path[-4:] != ".nx5"):
            file_path += ".nx5"
        h5w = h5py.File(file_path, 'w')
        #create the NeXuS/HDF5 tree
        gnm = 'FocalSeries:NXentry'
        h5w.create_group(gnm)
        gnm += '/Microscope:NXinstrument'
        h5w.create_group(gnm)
        h5w.create_dataset(gnm + '/Dx', data = np.float32(self.scaleX))
        h5w.create_dataset(gnm + '/Dy', data = np.float32(self.scaleY))
        h5w.create_dataset(gnm + '/DefStep', data = np.float32(self.DeltaFoc))
        h5w.create_dataset(gnm + '/DefStart', data = np.float32(self.centralSlice))
        h5w.create_dataset(gnm + '/Voltage_V', data = np.float32(self.highVoltage))
        h5w.create_dataset(gnm + '/Binning', data = np.float32(1.0))
        h5w.create_dataset(gnm + '/Alpha_mrad', data = np.float32(self.alpha))
        h5w.create_dataset(gnm + '/Delta_nm', data = np.float32(self.Delta))
        h5w.create_dataset(gnm + '/Cs_mm', data = np.float32(self.Cs))
        h5w.create_dataset(gnm + '/AstMag_nm', data = np.float32(self.astigMag))
        h5w.create_dataset(gnm + '/AstPhi_deg', data = np.float32(self.astigAngle))
        h5w.create_dataset(gnm + '/ObjApert_mrad', data = np.float32(self.apertRadius))
        h5w.create_dataset(gnm + '/ObjApertEdge_mrad', data = np.float32(self.apertEdge)) 
        h5w.create_dataset(gnm + '/ObjApertOffsetX_mrad', data = np.float32(self.apertOffsetX))
        h5w.create_dataset(gnm + '/ObjApertOffsetY_mrad', data = np.float32(self.apertOffsetY))
        h5w.create_dataset(gnm + '/DefocusStep_nm', data = np.float32(self.DeltaFoc))
        h5w.create_dataset(gnm + '/DefocusOffset_nm', data = np.float32(self.offsetFoc))
        h5w.create_dataset(gnm + '/DefocusExp', data = np.float32(self.focusExp))
        h5w.create_dataset(gnm + '/ROI', data = np.float32(dims))

        gnm = 'FocalSeries:NXentry' + '/DMPreprocessing:NXprocess'
        h5w.create_group(gnm)
        h5w.create_dataset(gnm + '/OffsetFoc', data = np.float32(self.offsetFoc))
        h5w.create_dataset(gnm + '/DefocusExp', data = np.float32(self.focusExp))
        
        rot = dict_get_value(self.metadata,"Rot_MuRad_mm")
        zoom = dict_get_value(self.metadata,"Zoomfactor_mm")
        if(rot):
            h5w.create_dataset(gnm + '/Rot_MuRad_mm', data = np.float32(rot))
        else:
            h5w.create_dataset(gnm + '/Rot_MuRad_mm', data = np.float32(0))
        if(zoom):
            h5w.create_dataset(gnm + '/ZoomFactor_mm', data = np.float32(zoom))
        else:
            h5w.create_dataset(gnm + '/ZoomFactor_mm', data = np.float32(1))
            
            

        #h5w.create_dataset(gnm + '/DataNormalized', data = np.float32(rootgrp.getncattr('DMtags.FRWR.normalized'))) ###MK::does not exist in *.cdf
        ###MK::likely a bug, but is dangerous because for all nc_get_attr calls there is no check of the retval so variables can
        #can remain unassigned silently !

        gnm = 'FocalSeries:NXentry' + '/data:NXdata'
        h5w.create_group(gnm)
        dst = h5w.create_dataset(gnm + '/ImageStack', data = np.float32(self.data))
        print(dst)
            
        h5w.close()
        
