import gettext
import asyncio
import logging
import numpy as np
import json
import h5py
import time

from nion.swift import DocumentController, Facade, Panel, Workspace
from nion.swift.model import PlugInManager, DataItem
from nion.typeshed import API_1_0
from nion.ui import Declarative
from nion.utils import Event, Registry
import nion.data.xdata_1_0 as xd

import nionswift_plugin.nionswift_frwr_plugin.FRWRUtils as FU

_ = gettext.gettext


class FRWRUIHandler:
    def __init__(self, api: API_1_0.API, event_loop: asyncio.AbstractEventLoop, ui_view: dict):
        self.ui_view = ui_view
        self.__api = api
        self.__event_loop = event_loop
        self.item_names = ["frwr_plugin_series","frwr_plugin_series_fft"]
        self.window = self.__api.application.document_windows[0] #application window
        self.document_controller = self.window._document_controller #nion.swift.DocumentController object. Most of the things not included in api can be realized with methods of this object
        self.ui = self.document_controller.ui
        self.data_item0 = None #data item containing the original data (isn't changed)
        self.data_item = None #plugin specific data item with just the defocus series images
        self.data_item_fft = None #plugin specific data item with the fft of the defocus series
        self.dimensional_calibrations = None #dimensional calibrations of the defocus images
        self.DefocusSetup = None #object handling all calculations of thon rings, defocus etc.
        
        #relates names of text input elemts to their respective variable in the DefocusSetup object
        self.text_edit_keys = {"setup_voltage": "highVoltage", "setup_cs": "Cs", "setup_n_defoci": "stepsFoc", "setup_defoc_step": "DeltaFoc", "setup_defoc_offset": "offsetFoc", "setup_exp_time": "expTime", "setup_astigmatism": "astigMag", "setup_angle": "astigAngle", "setup_delta": "Delta", "setup_alpha": "alpha", "setup_obj_apert": "apertRadius", "setup_edge": "apertEdge", "setup_offset_x": "apertOffsetX", "setup_offset_y": "apertOffsetY", "setup_ctf_rings": "CTFrings", "setup_ctf_points": "CTFpoints"}
        self.defocus_keys = ["apertRadius","offsetFoc","focusExp","DeltaFoc","Cs","astigMag","astigAngle","highVoltage","CTFrings","CTFpoints"]
        
        
    def extract_data_metadata(self):
        #decides whether a stack or multiple single images are selected as well as format of data (numpy array or hdf5 dataset (sometimes nion can't decide itself))
        #combines single images to a stack, transfering the metadata of the first image to the stack
        #metadata is formated such that generated nionswift related metadata is directly accessible by metadata["key"] whereas the actual loaded metadata is in metadata["metadata"]
        
        data_selected = self.document_controller.selected_data_items
        if (len(data_selected) == 1):
            self.data_item0 = self.window.target_data_item
            mtd = dict({})
            if (isinstance(self.data_item0.data, h5py._hl.dataset.Dataset)):
                mtd = json.loads(self.data_item0.data.attrs["properties"])
            elif (isinstance(self.data_item0.data, np.ndarray)):
                mtd['metadata'] = self.data_item0.metadata
                mtd['dimensional_calibrations'] = []
                for cal in self.data_item0.dimensional_calibrations:
                    cal_dic = {"offset":cal.offset,"scale": cal.scale,"units":cal.units}
                    mtd['dimensional_calibrations'].append(cal_dic)
            else:
                raise RuntimeError("Check data type")
            
            return self.data_item0.data[:,:,:], mtd
            
        elif (len(data_selected) > 1):
            self.data_item0 = data_selected[0]
            shape = data_selected[0].data.shape
            stack = np.zeros((len(data_selected),*shape))
            for i in range (0,len(data_selected)):
                slice_array = np.full(shape,np.mean(self.data_item0.data[:,:]))
                slice_data = data_selected[i].data[:,:]
                slice_array[0:min(shape[0],slice_data.shape[0]),0:min(shape[1],slice_data.shape[1])] = slice_data[0:min(shape[0],slice_data.shape[0]),0:min(shape[1],slice_data.shape[1])]
                stack[i] = slice_array
            
            mtd = dict({})
            if (isinstance(self.data_item0.data, h5py._hl.dataset.Dataset)):
                mtd = json.loads(self.data_item0.data.attrs["properties"])
            elif (isinstance(self.data_item0.data, np.ndarray)):
                mtd['metadata'] = self.data_item0.metadata
                mtd['dimensional_calibrations'] = [{"offset": 0,"scale": 1,"units": None}]
                for cal in self.data_item0.dimensional_calibrations:
                    cal_dic = {"offset":cal.offset,"scale": cal.scale,"units":cal.units}
                    mtd['dimensional_calibrations'].append(cal_dic)
            else:
                raise RuntimeError("Check data type")
            
            return stack,mtd
                
        else:
            raise RuntimeError("No data selected")
        
    def get_data_clicked(self, widget):
        #acquire data_item from targeted Display Panel
        
        stack,mtd = self.extract_data_metadata()
        self.DefocusSetup = FU.DefocusSetup(stack,mtd)
        
        self.setup_nonlinear_focus_check.checked = True
        self.setup_ring_mode.checked = True
        
        ''' find data items used to display the stack/fft or generate these '''
        data_item_flag = False
        data_item_fft_flag = False
        for item in self.__api.library.data_items:
            if (item.title == "frwr_plugin_series"):
                self.data_item = item
                self.data_item.data = self.DefocusSetup.slice_data[self.DefocusSetup.currentSlice]
                data_item_flag = True
            elif (item.title == "frwr_plugin_series_fft"):
                self.data_item_fft = item
                data_item_fft_flag = True
                self.data_item_fft.data = self.DefocusSetup.slice_data_fft[self.DefocusSetup.currentSlice]
        if (not data_item_flag):
            self.data_item = self.window.create_data_item_from_data(self.DefocusSetup.slice_data[self.DefocusSetup.currentSlice],title = "frwr_plugin_series")
        if (not data_item_fft_flag):
            self.data_item_fft = self.window.create_data_item_from_data(self.DefocusSetup.slice_data_fft[self.DefocusSetup.currentSlice],title = "frwr_plugin_series_fft")
            
        self.window.display_data_item(self.data_item)
        self.window.display_data_item(self.data_item_fft)
        
        ''' display thon rings of first slice '''
        coords = self.DefocusSetup.createThonRings(self.DefocusSetup.currentSlice)
        self.draw_annotations(coords)
        
        self.set_calibrations()
        
        ''' fill text inputs '''
        self.fill_info_tab()
        self.fill_text_edits()
        self.total_slice_label.text = "/" + str(self.DefocusSetup.stepsFoc-1)
        
    def set_calibrations(self):
        #gets the calibrations for the stack data item from metadata (done here and not in the DefocusSetup class, as that doesn't require offset or units)
        
        mtd_dim_cal = self.DefocusSetup.metadata["dimensional_calibrations"]
        i0 = self.DefocusSetup.dims_adj[0]
        i1 = self.DefocusSetup.dims_adj[1]
        dim_cal_0 = self.__api.create_calibration(mtd_dim_cal[i0]["offset"], mtd_dim_cal[i0]["scale"], mtd_dim_cal[i0]["units"])
        dim_cal_1 = self.__api.create_calibration(mtd_dim_cal[i1]["offset"], mtd_dim_cal[i1]["scale"], mtd_dim_cal[i1]["units"])
        self.dimensional_calibrations = [dim_cal_0, dim_cal_1]
        self.data_item.set_dimensional_calibrations(self.dimensional_calibrations)


    def fill_text_edits(self):
        for param in self.text_edit_keys:
            getattr(self, param).text = _(str(np.around(getattr(self.DefocusSetup,self.text_edit_keys[param]),2)))
        self.setup_nonlinear_focus_edit.text = _(str(self.DefocusSetup.focusExp))
        self.setup_defocus.text = _(str(np.around(self.DefocusSetup.defocusVal[self.DefocusSetup.currentSlice],2)))
        self.setup_slice.text = _(str(self.DefocusSetup.currentSlice))
        
    def fill_info_tab(self):
        self.info_spec_edit.text = _(str(FU.dict_get_value(self.DefocusSetup.metadata,"Specimen")))
        self.info_op_edit.text = _(str(FU.dict_get_value(self.DefocusSetup.metadata,"Operator")))
        self.info_mic_edit.text = _(str(FU.dict_get_value(self.DefocusSetup.metadata,"Microscope")))
        self.info_folder_edit.text = _(str(self.data_item0.title))
        self.info_series_edit.text = _(str(FU.dict_get_value(self.DefocusSetup.metadata,"seriesName")))
        self.info_series_ind_edit.text = _(str(FU.dict_get_value(self.DefocusSetup.metadata,"seriesIndex")))
        
    

    def param_changed(self,widget,text):
        for param in self.text_edit_keys:
            key = self.text_edit_keys[param]
            if(widget == getattr(self,param)):
                if (text != str(np.around(getattr(self.DefocusSetup,key),2))):
                    try:
                        if (key not in ["CTFrings","CTFpoints"]):
                            setattr(self.DefocusSetup,key,float(text))
                        else:
                            setattr(self.DefocusSetup,key,int(text))
                        if(key in self.defocus_keys):
                            self.DefocusSetup.draw_apert()
                            coords = self.DefocusSetup.createThonRings(self.DefocusSetup.currentSlice)
                            self.draw_annotations(coords)
                            self.data_item_fft.data = self.DefocusSetup.slice_data_fft[self.DefocusSetup.currentSlice]
                            self.fill_text_edits()
                    except:
                        widget.text = _(str(np.around(getattr(self.DefocusSetup,key),2)))
                break
                
    def focusExp_changed(self,widget,text):
        if(text != str(np.around(self.DefocusSetup.focusExp,2))):
            try:
                self.DefocusSetup.focusExp = float(text)
                coords = self.DefocusSetup.createThonRings(self.DefocusSetup.currentSlice)
                self.draw_annotations(coords)
                self.fill_text_edits()
            except:
                widget.text = _(str(self.DefocusSetup.focusExp))
                
    def defocus_changed(self,widget,text):
        #called if defocus is manually set
        #new defocus value is then passed to the DefocusSetup object
        
        if (text != str(np.around(self.DefocusSetup.defocusVal[self.DefocusSetup.currentSlice],2))):
            try:
                self.DefocusSetup.update_defocus(float(text))
                coords = self.DefocusSetup.createThonRings(self.DefocusSetup.currentSlice)
                self.draw_annotations(coords)
                self.fill_text_edits()
            except Exception as e:
                logging.error(e)
                self.setup_defocus.text = _(str(np.around(self.DefocusSetup.defocusVal[self.DefocusSetup.currentSlice],2)))
    
    def defocus_button_clicked(self,widget):
        #changes defocus with the up down buttons
        
        def_change = np.floor(np.log(abs(float(self.setup_defocus.text))/10)/(np.log(10)))
        newdef = self.DefocusSetup.defocusVal[self.DefocusSetup.currentSlice]
        if(widget == self.def_up):
            newdef += max(10**def_change,10)
        else:
            newdef -= max(10**def_change,10)
        self.setup_defocus.text = _(str(np.around(newdef,2)))
        self.defocus_changed(self.setup_defocus,str(newdef))

    '''def update_selection(self, widget):
        self.data_item_fft.display.add_graphic({"type":"ellipse-graphic","center":(0.75,0.5),"size":(0.5,0.5),"stroke_color" : "red","is_position_locked" : True, "is_shape_locked" : True})
        pass
    '''
    
    def change_slice(self, widget):
        #change slice with arrows
        
        if (widget == self.slice_down):
            self.DefocusSetup.currentSlice -= 1
            self.DefocusSetup.currentSlice = max(self.DefocusSetup.currentSlice,0)
        else:
            self.DefocusSetup.currentSlice += 1
            self.DefocusSetup.currentSlice = min(self.DefocusSetup.currentSlice,self.DefocusSetup.stepsFoc-1)
        
        coords = self.DefocusSetup.createThonRings(self.DefocusSetup.currentSlice)
        self.draw_annotations(coords)
        self.data_item.data = self.DefocusSetup.slice_data[self.DefocusSetup.currentSlice]
        self.data_item.set_dimensional_calibrations(self.dimensional_calibrations)
        self.data_item_fft.data = self.DefocusSetup.slice_data_fft[self.DefocusSetup.currentSlice]
        self.setup_slice.text = _(str(self.DefocusSetup.currentSlice))
        self.setup_defocus.text = _(str(np.around(self.DefocusSetup.defocusVal[self.DefocusSetup.currentSlice],2)))
        
    def set_slice(self,widget,text):
        #change slice through text input directly
        
        try:
            index = int(text)
            index = min(max(index,0),self.DefocusSetup.stepsFoc-1)
            self.DefocusSetup.currentSlice = index
            coords = self.DefocusSetup.createThonRings(self.DefocusSetup.currentSlice)
            self.draw_annotations(coords)
            self.data_item.data = self.DefocusSetup.slice_data[self.DefocusSetup.currentSlice]
            self.data_item.set_dimensional_calibrations(self.dimensional_calibrations)
            self.data_item_fft.data = self.DefocusSetup.slice_data_fft[self.DefocusSetup.currentSlice]
            widget.text = _(str(self.DefocusSetup.currentSlice))
            self.setup_defocus.text = _(str(np.around(self.DefocusSetup.defocusVal[self.DefocusSetup.currentSlice],2)))
        except:
            widget.text = _(str(self.DefocusSetup.currentSlice))
        
    def check_changed(self,widget,checked):
        if(widget == self.setup_nonlinear_focus_check):
            self.DefocusSetup.nonLinearFlag = checked
            coords = self.DefocusSetup.createThonRings(self.DefocusSetup.currentSlice)
            self.draw_annotations(coords)
            self.data_item_fft.data = self.DefocusSetup.slice_data_fft[self.DefocusSetup.currentSlice]
            self.fill_text_edits()
        elif(widget == self.setup_ring_mode):
            self.DefocusSetup.RingFlag = checked
            coords = self.DefocusSetup.createThonRings(self.DefocusSetup.currentSlice)
            self.draw_annotations(coords)
            
    def save_clicked(self,widget):
        if (len(self.data_item.graphics) != 0):
            self.debug_label.text=_("")
            dims = np.float32(self.data_item.graphics[0].bounds)
            dims = dims.flatten()
            file_path, f_filter, directory = self.ui.get_save_file_path('Save As...', '~/'+str(self.data_item0.title)+'_setup.nx5', 'hdf5 File (*.nx5);;All Files (*.*)')
            if (file_path != ""):
                self.DefocusSetup.save(dims,file_path)
        else:
            self.debug_label.text=_("Please select a ROI in 'frwr_plugin_series' before saving!")
        
    def draw_annotations(self,coords):
        rad = 10
        graphics = self.data_item_fft.graphics
        nrings = self.DefocusSetup.CTFrings
        npoints = 1
        annotCount = 0
        shift = 0
        
        if (self.DefocusSetup.RingFlag):
            annotCount = len(graphics)-nrings
        else:
            npoints = self.DefocusSetup.CTFpoints
            annotCount = len(graphics)-nrings*npoints
        ''' removes or adds additional annotations in accordance with annotCount '''
        if (annotCount > 0):
            for i in range(0,annotCount):
                self.data_item_fft.remove_region(graphics[i])
            shift = -annotCount
        else:
            shift = -annotCount
            for i in range(0,-annotCount):
                i1 = i//npoints
                i2 = i%npoints
                self.data_item_fft.display.add_graphic({"type":"ellipse-graphic","center":(coords[i1,i2,1]/self.DefocusSetup.height,coords[i1,i2,0]/self.DefocusSetup.width),"size":(2*coords[i1,i2,3]/self.DefocusSetup.height,2*coords[i1,i2,2]/self.DefocusSetup.width),"rotation": coords[i1,i2,4],"stroke_color" : "red","is_position_locked" : True, "is_shape_locked" : True})
                annotCount = 0
        ''' moves annotations to their correct locations '''
        for k in range(annotCount,len(graphics)):
            i = k+shift
            i1 = i//npoints
            i2 = i%npoints
            graphic = graphics[k]
            graphic.center = (coords[i1,i2,1]/self.DefocusSetup.height,coords[i1,i2,0]/self.DefocusSetup.width)
            graphic.size = (2*coords[i1,i2,3]/self.DefocusSetup.height,2*coords[i1,i2,2]/self.DefocusSetup.width)
            graphic.set_property('rotation',coords[i1,i2,4])
        
    def close(self):
        pass
        


class FRWRUI(object):
    def __init__(self):
        self.panel_type = 'frwr-panel'
        
    def get_ui_handler(self, api_broker: PlugInManager.APIBroker=None, event_loop: asyncio.AbstractEventLoop=None, **kwargs):
        api = api_broker.get_api('~1.0')
        ui = api_broker.get_ui('~1.0')
        ui_view = self.__create_ui_view(ui)
        return FRWRUIHandler(api, event_loop, ui_view)
    
    def __create_ui_view(self, ui: Declarative.DeclarativeUI) -> dict:
        #info_tab
        #Specimen
        info_spec_label = ui.create_label(text = _("Specimen:"))
        info_spec_edit = ui.create_line_edit(name  = "info_spec_edit")
        info_spec_row = ui.create_row(info_spec_label, ui.create_spacing(26), info_spec_edit, name = "info_spec_row", spacing = 10)
        #Operator
        info_op_label = ui.create_label(text = _("Operator:"))
        info_op_edit = ui.create_line_edit(name  = "info_op_edit")
        info_op_row = ui.create_row(info_op_label, ui.create_spacing(29), info_op_edit, name = "info_op_row", spacing = 10)
        #Microscope
        info_mic_label = ui.create_label(text = _("Microscope:"))
        info_mic_edit = ui.create_line_edit(name  = "info_mic_edit")
        info_mic_row = ui.create_row(info_mic_label, ui.create_spacing(14), info_mic_edit, name = "info_mic_row", spacing = 10)
        #Folder
        info_folder_label = ui.create_label(text = _("Folder:"))
        info_folder_edit = ui.create_line_edit(name  = "info_folder_edit")
        info_folder_row = ui.create_row(info_folder_label, ui.create_spacing(48), info_folder_edit, name = "info_folder_row", spacing = 10)
        #Series name
        info_series_label = ui.create_label(text = _("Series name:"))
        info_series_edit = ui.create_line_edit(name  = "info_series_edit")
        info_series_ind_label = ui.create_label(text = _("Index:"))
        info_series_ind_edit = ui.create_line_edit(name  = "info_series_ind_edit", width = 50)
        info_series_row = ui.create_row(info_series_label, ui.create_spacing(10), info_series_edit,info_series_ind_label, info_series_ind_edit, name = "info_series_row", spacing = 10)
        #get Data
        info_get_data_button = ui.create_push_button(name = "info_get_data_button", text=_("load target data item"), on_clicked="get_data_clicked", widget_id = "test_id")
        #info_update_button = ui.create_push_button(name = "info_update_button", text=_("test button"), on_clicked="update_selection")
        info_get_data_row = ui.create_row(info_get_data_button, name="info_get_data_row", spacing=10)
        #main
        info_main_column = ui.create_column(info_spec_row, info_op_row, info_mic_row, info_folder_row, info_series_row, info_get_data_row,ui.create_stretch(), name = "info_main_column",spacing = 8, margin = 10)
        
        #setup tab
        #parameter text inputs
        setup_dicts = [[ _("HighVoltage (kV):"),"setup_voltage"], [_("Cs (mm):"),"setup_cs"], [_("Number of defoci:"),"setup_n_defoci"], [_("Defocus step:"),"setup_defoc_step"], [_("Defocus offset:"), "setup_defoc_offset"], [_("Exposure time:"), "setup_exp_time"], [_("Astigmatism (mm):"), "setup_astigmatism"], [_("Angle (deg):"), "setup_angle"], [_("Delta (nm):"), "setup_delta"], [_("alpha (mrad):"), "setup_alpha"], [_("Obj. apert. (mrad):"), "setup_obj_apert"], [_("Edge (fract.):"), "setup_edge"], [_("Offset X (mrad):"), "setup_offset_x"], [_("Offset Y (mrad):"), "setup_offset_y"], [_("CTF-rings:"), "setup_ctf_rings"], [_("CTF-points"), "setup_ctf_points"]]
        setup_row_list = []
        
        for i in range(0,len(setup_dicts),2):
            edit1 = [ui.create_label(text = setup_dicts[i][0]),ui.create_line_edit(name = setup_dicts[i][1], width=70, on_editing_finished = "param_changed")]
            edit2 = [ui.create_label(text = setup_dicts[i+1][0]),ui.create_line_edit(name = setup_dicts[i+1][1], width=70, on_editing_finished = "param_changed")]
            row_param = [*edit1,*edit2]
            setup_row_list.append(ui.create_row(*row_param,spacing = 10))
            
        #checkbox row
        checkbox_param = [ui.create_check_box(text="Ring mode", name="setup_ring_mode", check_state = "checked", on_checked_changed="check_changed"),ui.create_check_box(text=_("Non-linear Focus"), name="setup_nonlinear_focus_check", check_state = "checked", on_checked_changed="check_changed"), ui.create_line_edit(name = "setup_nonlinear_focus_edit", width = 70, on_editing_finished = "focusExp_changed")]
        setup_checkbox_row = ui.create_row(ui.create_stretch(), *checkbox_param, ui.create_stretch(),spacing = 10)
        
        #slice row
        def_ud = ui.create_column(ui.create_push_button(name = "def_up", text = _("↑"), on_clicked = "defocus_button_clicked", width = 18, height = 17), ui.create_push_button(name = "def_down", text = _("↓"), on_clicked = "defocus_button_clicked", width = 18, height = 17), spacing = 0)
        spc = ui.create_spacing(20)
        
        slice_param = [ui.create_push_button(name = "slice_down", text = _("<--"), on_clicked = "change_slice", width = 50),spc, ui.create_label(name = "slice_label", text = _("Slice:")),spc, ui.create_line_edit(name = "setup_slice",width= 40, on_editing_finished="set_slice"),ui.create_spacing(10), ui.create_label(name = "total_slice_label", text = _("/0")),spc, ui.create_spacing(30), ui.create_label(text = _("Defocus:")),spc, ui.create_spacing(10), ui.create_line_edit(name = "setup_defocus", width = 90, on_editing_finished = "defocus_changed"), def_ud,spc, ui.create_label(text = _("nm")),spc, ui.create_spacing(20),ui.create_push_button(name = "slice_up", text = _("-->"), on_clicked = "change_slice", width = 50)]
        
        setup_slice_row = ui.create_row(ui.create_stretch(), *slice_param, ui.create_stretch(), spacing = 0)
        
        save_row = ui.create_row(ui.create_stretch(),ui.create_push_button(name = "save_button", text = "save", on_clicked = "save_clicked"),ui.create_stretch(),spacing = 0)
        debug_row = ui.create_row(ui.create_stretch(),ui.create_label(text="",name="debug_label"),ui.create_stretch())
        
        setup_main_column = ui.create_column(*setup_row_list,setup_checkbox_row,setup_slice_row, save_row,debug_row, ui.create_stretch(), spacing = 8, margin = 10)
        
        #plugin tabs
        info_tab = ui.create_tab(content = info_main_column, label = _("Info"))
        setup_tab = ui.create_tab(content = setup_main_column, label = _("Focal Series Project Setup"))
        frwr_tabs = ui.create_tabs(info_tab, setup_tab, name = "frwr_tabs")
        
        main_stack = ui.create_stack(frwr_tabs,name = "main_stack")
        return main_stack
        


class FRWRPanel(Panel.Panel):
    def __init__(self, document_controller: DocumentController.DocumentController, panel_id: str, properties: dict):
        super().__init__(document_controller, panel_id, 'frwr-panel')
        panel_type = properties.get('panel_type')
        for component in Registry.get_components_by_type('frwr-panel'):
            if component.panel_type == panel_type:
                ui_handler = component.get_ui_handler(api_broker=PlugInManager.APIBroker(), event_loop=document_controller.event_loop)
                self.widget = Declarative.DeclarativeWidget(document_controller.ui, document_controller.event_loop, ui_handler)
                
                


def run():
    Registry.register_component(FRWRUI(), {'frwr-panel'})
    panel_properties = {'panel_type': 'frwr-panel'}
    Workspace.WorkspaceManager().register_panel(FRWRPanel, 'frwr-control-panel', _('FRWR'), ['left', 'right'], 'left', panel_properties)
