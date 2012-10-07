# -*- coding: utf-8 -*-
import models.sxrd_test5_sym_new_test_new66_2 as model
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv
import sys
sys.path.append('\\home\\tlab\\Desktop\\genx2.0.0\\polyhedra-geometry')
import hexahedra,hexahedra_distortion,tetrahedra,octahedra,tetrahedra_edge_distortion,trigonal_pyramid_distortion,trigonal_pyramid_distortion_shareface,trigonal_pyramid_distortion2,trigonal_pyramid_distortion3,trigonal_pyramid_distortion4

class domain_creator():
    def __init__(self,ref_domain,id_list,terminated_layer=0,domain_N=1,new_var_module=None,z_shift=0.):
        #id_list is a list of id in the order of ref_domain,terminated_layer is the index number of layer to be considered
        #for termination,domain_N is a index number for this specific domain, new_var_module is a UserVars module to be used in
        #function of set_new_vars
        self.ref_domain=ref_domain
        self.id_list=id_list
        self.terminated_layer=terminated_layer
        self.domain_N=domain_N
        self.share_face,self.share_edge,self.share_corner=(False,False,False)
        #self.anchor_list=[]
        self.polyhedra_list=[]
        self.new_var_module=new_var_module
        self.z_shift=z_shift
        self.domain_A,self.domain_B=self.create_equivalent_domains()
    
    def build_super_cell(self,ref_domain,rem_atom_ids=None):
    #build a super cell based on the ref_domain, the super cell is actually two domains stacking together in x direction
    #rem_atom_ids is a list of atom ids you want to remove before building a super cell
        super_cell=ref_domain.copy()
        if rem_atom_ids!=None:
            for i in rem_atom_ids:
                super_cell.del_atom(i)
                
        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
            
        for id in super_cell.id:
            index=np.where(ref_domain.id==id)[0][0]
            super_cell.add_atom(id=str(id)+'_+x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1], z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0], y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_+x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]+1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x+y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]+1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
            super_cell.add_atom(id=str(id)+'_-x-y',element=ref_domain.el[index], x=_extract_coor(ref_domain,id)[0]-1.0, y=_extract_coor(ref_domain,id)[1]-1., z=_extract_coor(ref_domain,id)[2], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
        
        return super_cell
    
    def create_equivalent_domains(self):
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #number 5 here is crystal specific, here is the case for hematite
        for id in self.id_list[:(self.terminated_layer+5)*2]:
            new_domain_B.del_atom(id)
        return new_domain_A,new_domain_B
        
    def add_sorbate_polyhedra(self,domain,r=0.1,theta=[0.,0.],phi=[np.pi/2,np.pi/2],polyhedra_flag='tetrahedra',\
            extra_flag='1_1+0_1',extra_flag2='type1',attach_atm_id=[['id1','id2'],['id3','id4']],offset=[[None,None],[None,None]],el='Pb',id_attach=[],use_ref=False):
        #note the offset symbol means calculating the attached site in the adjacent unit cell
        #'+x'means plus 1 for x direction, '-x' means minus 1 for x direction and so on for '+y'and '-y'
        #theta and phi is list with at most two items, when considering share-corner, and  when considering
        #shareing edge, theta and phi list contain only one item. extra flags has values depending on the polyhedra
        #type used, attach atm id (oxygen id at the surface) is a list of list, the list inside has items of one (share corner), two (share edge)
        #or three (share face),id attach has the same length as attach atm id, each item is a str symbol used to distinguish
        #the added Pb,for say, and O, is one of the item is 'A', then the associated ids for the added atoms will be like
        #Pb_A, Os_A_0,Os_A_1. The number of id attach is the number of Pb types added. And you should see the relationship b/
        #id of Pb and O, so Pb_A will have Os_A_n like oxygen attached (same A),and Pb_AA will have Os_AA_n like oxygen attached (same AA)
        #this function will add several types of Pb at the surface,each type will correspoind to a polyhedra in self.polyhedra_list
        #you shoul know the index of chemically equivalent polyhedra, it should be every other number,like 0 and 2,or 1 and 3.
        
        N_vertices=len(attach_atm_id[0])
        if N_vertices==3:self.share_face=True
        elif N_vertices==2:self.share_edge=True
        elif N_vertices==1:self.share_corner=True
        for i in range(len(attach_atm_id)):
            anchor=np.array([[0.,0.,0.]])
            for j in range(N_vertices):
                index=np.where(domain.id==attach_atm_id[i][j])[0][0]
                pt_x,pt_y,pt_z=domain.x[index],domain.y[index],domain.z[index]
                if offset[i][j]=='+x':
                    anchor=np.append(anchor,np.array([[(pt_x+1.)*5.038,pt_y*5.434,pt_z*7.3707]]),axis=0)
                elif offset[i][j]=='-x':
                    anchor=np.append(anchor,np.array([[(pt_x-1.)*5.038,pt_y*5.434,pt_z*7.3707]]),axis=0)
                elif offset[i][j]=='+y':
                    anchor=np.append(anchor,np.array([[pt_x*5.038,(pt_y+1.)*5.434,pt_z*7.3707]]),axis=0)
                elif offset[i][j]=='-y':
                    anchor=np.append(anchor,np.array([[pt_x*5.038,(pt_y-1.)*5.434,pt_z*7.3707]]),axis=0)
                else:
                    anchor=np.append(anchor,np.array([[pt_x*5.038,pt_y*5.434,pt_z*7.3707]]),axis=0)
            anchor=anchor[1::]
            oxygens=np.array([[0.,0.,0.]])
            polyhedra=0
            if polyhedra_flag=='tetrahedra':
                if N_vertices==3:
                    polyhedra=tetrahedra.share_face(face=anchor)
                    polyhedra.share_face_init()
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+3)) for ii in range(1)],axis=0)[1::]
                elif N_vertices==2:
                    polyhedra=tetrahedra.share_edge(edge=anchor)
                    if use_ref==True:
                        ref_p=(anchor[0]+anchor[1])/2+np.array([0,0.,2.])
                        polyhedra.cal_p2(ref_p=ref_p,theta=theta[0],phi=phi[0])
                    else:polyhedra.cal_p2(theta=theta[0],phi=phi[0])
                    polyhedra.share_face_init()
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+2)) for ii in range(2)],axis=0)[1::]
                elif N_vertices==1:
                    polyhedra=tetrahedra.share_corner(corner=anchor[0])
                    polyhedra.cal_p1(r=r,theta=theta[0],phi=phi[0])
                    polyhedra.cal_p2(theta=theta[1],phi=phi[1])
                    polyhedra.share_face_init()
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+1)) for ii in range(3)],axis=0)[1::]
            elif polyhedra_flag=='hexahedra':
                if N_vertices==3:
                    polyhedra=hexahedra.share_face(face=anchor)
                    polyhedra.share_face_init(flag=extra_flag)
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+3)) for ii in range(2)],axis=0)[1::]
                elif N_vertices==2:
                    polyhedra=hexahedra.share_edge(edge=anchor)
                    if use_ref==True:
                        ref_p=(anchor[0]+anchor[1])/2+np.array([0,0.,2.])
                        polyhedra.cal_p2(ref_p=ref_p,theta=theta[0],phi=phi[0],flag=extra_flag,extend_flag=extra_flag2)
                    else:polyhedra.cal_p2(theta=theta[0],phi=phi[0],flag=extra_flag,extend_flag=extra_flag2)
                    polyhedra.share_face_init(polyhedra.flag)
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+2)) for ii in range(3)],axis=0)[1::]
                elif N_vertices==1:
                    polyhedra=hexahedra.share_corner(corner=anchor[0])
                    polyhedra.cal_p1(r=r,theta=theta[0],phi=phi[0])
                    polyhedra.cal_p2(theta=theta[1],phi=phi[1],flag=extra_flag,extend_flag=extra_flag2)
                    polyhedra.share_face_init(polyhedra.flag)
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+1)) for ii in range(4)],axis=0)[1::]
            elif polyhedra_flag=='octahedra':
                if N_vertices==3:
                    polyhedra=octahedra.share_face(face=anchor)
                    polyhedra.share_face_init(flag=extra_flag)
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+3)) for ii in range(3)],axis=0)[1::]
                elif N_vertices==2:
                    polyhedra=octahedra.share_edge(edge=anchor)
                    if use_ref==True:
                        ref_p=(anchor[0]+anchor[1])/2+np.array([0,0.,2.])
                        polyhedra.cal_p2(ref_p=ref_p,theta=theta[0],phi=phi[0],flag=extra_flag)
                    else:polyhedra.cal_p2(theta=theta[0],phi=phi[0],flag=extra_flag)
                    polyhedra.share_face_init(polyhedra.flag)
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+2)) for ii in range(4)],axis=0)[1::]
                elif N_vertices==1:
                    polyhedra=octahedra.share_corner(corner=anchor[0])
                    polyhedra.cal_p1(r=r,theta=theta[0],phi=phi[0])
                    polyhedra.cal_p2(theta=theta[1],phi=phi[1],flag=extra_flag)
                    polyhedra.share_face_init(polyhedra.flag)
                    self.polyhedra_list.append(polyhedra)
                    oxygens=np.append(oxygens,[getattr(polyhedra,'p'+str(ii+1)) for ii in range(5)],axis=0)[1::]
            #Pb is at body center, which is the center point here
            domain.add_atom(id=el+'_'+id_attach[i],element=el,x=polyhedra.center_point[0]/5.038,y=polyhedra.center_point[1]/5.434,z=polyhedra.center_point[2]/7.3707,u=1.)
            for iii in range(len(oxygens)):
                o_xyz=oxygens[iii,:]
                domain.add_atom(id='Os_'+id_attach[i]+'_'+str(iii),element='O',x=o_xyz[0]/5.038,y=o_xyz[1]/5.434,z=o_xyz[2]/7.3707,u=0.32)

    def updata_polyhedra_orientation(self,polyhedra_index=[0,2],r=None,phi_list=[],theta_list=[],flag1='0_2+0_1',flag2='type1'):
        #this function will change T matrix and center point for each polyhedra
        #actually we want to change the coordinate system by rotating over the shared corner or edge
        #it won't change the atom position by now, the seting of the phi and theta list and the flags depend on polyhedra used
        for i in polyhedra_index:
            if self.share_corner==True:
                self.polyhedra_list[i].cal_p1(r=r,theta=theta_list[0],phi=phi_list[0])
                #print self.polyhedra_list[i].edge
                self.polyhedra_list[i].cal_p2(theta=theta_list[1],phi=phi_list[1],flag=flag1,extend_flag=flag2)
                self.polyhedra_list[i].share_face_init(flag=self.polyhedra_list[i].flag)
            elif self.share_edge==True:
                self.polyhedra_list[i].cal_p2(theta=theta_list[0],phi=phi_list[0],flag=flag1,extend_flag=flag2)
                self.polyhedra_list[i].share_face_init(flag=self.polyhedra_list[i].flag)
                
    def updata_polyhedra_orientation_batch(self,file):
        def norm_phi(phi):
            normalized_phi=0
            if phi<=0.5:
                normalized_phi=1.5708
            elif phi>0.5:
                normalized_phi=4.7124
            return normalized_phi
        f=open(file)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                polyhedra_index=0.
                try:
                    polyhedra_index=[int(line_split[0]),int(line_split[1])]
                except:
                    polyhedra_index=[int(line_split[0])]
                r=0
                try:
                    r=getattr(self.new_var_module,line_split[2])
                except:
                    pass
                list_len=int(line_split[3])
                theta_list=[getattr(self.new_var_module,i) for i in line_split[4:4+list_len]]
                phi_list=[norm_phi(getattr(self.new_var_module,i)) for i in line_split[4+list_len:4+2*list_len]]
                flag1=line_split[-3]
                flag2=line_split[-2]
                self.updata_polyhedra_orientation(polyhedra_index,r,phi_list,theta_list,flag1,flag2)
        f.close()
        
    def updata_polyhedra_center_point(self,domain_list,Pb_id_list,polyhedra_index_list):
        #this function will change the position of body center during fitting
        #all list has the same dimension
        for i in range(len(domain_list)):
            index=list(domain_list[i].id).index(Pb_id_list[i])
            x=5.038*(domain_list[i].x[index]+domain_list[i].dx1[index]+domain_list[i].dx2[index]+domain_list[i].dx3[index])
            y=5.434*(domain_list[i].y[index]+domain_list[i].dy1[index]+domain_list[i].dy2[index]+domain_list[i].dy3[index])
            z=7.3707*(domain_list[i].z[index]+domain_list[i].dz1[index]+domain_list[i].dz2[index]+domain_list[i].dz3[index])
            self.polyhedra_list[polyhedra_index_list[i]].center_point=np.array([x,y,z])
    
    def updata_polyhedra_center_point_batch(self,file):
        f=open(file)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                n=(len(line_split)-1)/3
                domain_list=[vars(self)[line_split[0+i]] for i in range(n)]
                Pb_id_list=[line_split[n+i] for i in range(n)]
                polyhedra_index_list=[int(line_split[n+n+i]) for i in range(n)]
                self.updata_polyhedra_center_point(domain_list,Pb_id_list,polyhedra_index_list)
        f.close()
        
    def adding_pb_share_triple(self,domain,attach_atm_ids=['id1','id2','id3'],offset=[None,None,None],pb_id='pb_id'):
        #the pb will be placed in a plane determined by three points,and lead position is equally distant from the three points
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        p_O3_index=np.where(domain.id==attach_atm_ids[2])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        #try to calculate the center point on the plane, two linear equations based on distance equivalence,one based on point on the plane
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        p_O3=pt_ct(domain,p_O2_index,offset[2])
        p0,p1,p2=p_O1,p_O2,p_O3
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])
        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)
        sorbate_v=center_point
        
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        #return anstrom
        return sorbate_v*basis
        
    def adding_pb_share_triple2(self,domain,r,attach_atm_ids=['id1','id2','id3'],offset=[None,None,None,None],bodycenter_id='Fe',pb_id='pb_id'):
        #the pb will be placed at a point starting from body center of the knonw polyhedra and through a center of a plane determined by three specified points,and lead will be placed somewhere on the extention line
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        p_O3_index=np.where(domain.id==attach_atm_ids[2])
        basis=np.array([5.038,5.434,7.3707])
        body_center_index=np.where(domain.id==bodycenter_id)
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        #try to calculate the center point on the plane, two linear equations based on distance equivalence,one based on point on the plane
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        p_O3=pt_ct(domain,p_O3_index,offset[2])
        p0,p1,p2=p_O1,p_O2,p_O3
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])
        #print p_O1,p_O2,p_O3
        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)
        #sorbate_v=center_point
        
        body_center=pt_ct(domain,body_center_index,offset[3])
        v_bc_fc=(center_point-body_center)*basis
        d_bc_fc=f2(center_point*basis,body_center*basis)
        scalor=(r+d_bc_fc)/d_bc_fc
        sorbate_v=(v_bc_fc*scalor+body_center*basis)/basis
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        #return anstrom
        return sorbate_v*basis
        
    def adding_pb_share_triple3(self,domain,r,off_constant=1.0,attach_atm_ids=['id1','id2','id3'],offset=[None,None,None,None],bodycenter_id='Fe',pb_id='pb_id'):
        #the pb will be placed at a point starting from body center of the knonw polyhedra and through a center of a plane determined by three specified points,and lead will be placed somewhere on the extention line
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        p_O3_index=np.where(domain.id==attach_atm_ids[2])
        basis=np.array([5.038,5.434,7.3707])
        body_center_index=np.where(domain.id==bodycenter_id)
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        #try to calculate the center point on the plane, two linear equations based on distance equivalence,one based on point on the plane
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        p_O3=pt_ct(domain,p_O3_index,offset[2])
        p0,p1,p2=p_O1,p_O2,p_O3
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])
        #print p_O1,p_O2,p_O3
        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)
        #sorbate_v=center_point
        v_p0_fc=center_point-p0
        center_point_new=v_p0_fc*off_constant+p0
        body_center=pt_ct(domain,body_center_index,offset[3])
        
        v_bc_fc=(center_point_new-body_center)*basis
        d_bc_fc=f2(center_point_new*basis,body_center*basis)
        scalor=(r+d_bc_fc)/d_bc_fc
        sorbate_v=(v_bc_fc*scalor+body_center*basis)/basis
        
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        #return anstrom
        return sorbate_v*basis
        
    def adding_pb_share_triple4(self,domain,r,attach_atm_ids=['id1','id2','id3'],offset=[None,None,None],pb_id='pb_id'):
        #similar to adding_pb_share_triple2, but no body center, the center point on the plane determined by attach atoms will be the starting point1
        #the pb will be added on the extention line of normal vector (normal to the plane) starting at starting point
        #the distance bt pb and the plane is specified by r, which is in unit of angstrom
        
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        p_O3_index=np.where(domain.id==attach_atm_ids[2])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        #try to calculate the center point on the plane, two linear equations based on distance equivalence,one based on point on the plane
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        p_O3=pt_ct(domain,p_O3_index,offset[2])
        p0,p1,p2=p_O1,p_O2,p_O3
        normal=np.cross(p1-p0,p2-p0)
        c3=np.sum(normal*p0)
        A=np.array([2*(p1-p0),2*(p2-p0),normal])

        C=np.array([np.sum(p1**2-p0**2),np.sum(p2**2-p0**2),c3])
        center_point=np.dot(inv(A),C)

        normal_scaled=r/(np.dot(normal*basis,normal*basis)**0.5)*normal
        sorbate_v=normal_scaled+center_point
        
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        #return anstrom
        return sorbate_v*basis
        
    def adding_pb_shareedge(self,domain,r=2.,attach_atm_ids=['id1','id2'],offset=[None,None,None],bodycenter_id='Fe',pb_id='pb_id'):
        #the pb will be placed on the extension line from rooting from bodycenter trough edge center
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        body_center_index=np.where(domain.id==bodycenter_id)
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])
        p_O2=pt_ct(domain,p_O2_index,offset[1])
        body_center=pt_ct(domain,body_center_index,offset[2])
        
        p1p2_center=(p_O1+p_O2)/2.
        v_bc_ec=(p1p2_center-body_center)*basis
        d_bc_ec=f2(body_center*basis,p1p2_center*basis)
        scalor=(r+d_bc_ec)/d_bc_ec
        sorbate_v=(v_bc_ec*scalor+body_center*basis)/basis
        sorbate_index=None
        try:
            sorbate_index=np.where(domain.id==pb_id)[0][0]
        except:
            domain.add_atom( pb_id, "Pb",  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
        if sorbate_index!=None:
            domain.x[sorbate_index]=sorbate_v[0]
            domain.y[sorbate_index]=sorbate_v[1]
            domain.z[sorbate_index]=sorbate_v[2]
        return sorbate_v*basis
        
    def adding_pb_shareedge_edge_distortion(self,domain,edge_offset=[0.,0.,0.],switch=False,phi=0.,attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1','id2']):
        #The added sorbates (including Pb and Os) will form a edge-distorted tetrahedra configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        tetrahedra_distortion=tetrahedra_edge_distortion.tetrahedra_edge_distortion(p0=p_O1,p1=p_O2,offset=edge_offset)
        tetrahedra_distortion.all_in_all(switch=switch,phi=phi)
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=tetrahedra_distortion.body_center/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=tetrahedra_distortion.p2/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[1],el='O',sorbate_v=tetrahedra_distortion.p3/basis)

    def adding_sorbate_pyramid_distortion(self,domain,edge_offset=[0.,0.],top_angle=1.,switch=False,phi=0.,attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1']):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        pyramid_distortion=trigonal_pyramid_distortion.trigonal_pyramid_distortion(p0=p_O1,p1=p_O2,top_angle=top_angle,len_offset=edge_offset)
        pyramid_distortion.all_in_all(switch=switch,phi=phi)
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=pyramid_distortion.p2/basis)
    
    def adding_sorbate_pyramid_distortion_B(self,domain,top_angle=1.,attach_atm_ids_ref=['id1','id2'],attach_atm_id_third=['id3'],offset=[None,None,None],pb_id='pb_id'):
        #here only consider the angle distortion specified by top_angle (range from 0 to 120 dg), and no length distortion, so the base is a equilayer triangle
        #and here consider the tridentate complexation configuration 
        #two steps:
        #step one: use the coors of the two reference atoms, and the third one to calculate a new third one such that this new point and 
        #the two ref pionts form a equilayer triangle, and the distance from third O to the new third one is shortest
        #see function of _cal_coor_o3 for detail
        #then based on these three points and top angle, calculate the apex coords
        #step two: 
        #update the coord of the third oxygen to the new third coords (be carefule about the offset, you must consider the coor within the unitcell)
        p_O1_index=np.where(domain.id==attach_atm_ids_ref[0])
        p_O2_index=np.where(domain.id==attach_atm_ids_ref[1])
        p_O3_index=np.where(domain.id==attach_atm_id_third[0])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        
        pt_ct2=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0],\
                       domain.y[p_O1_index][0],\
                       domain.z[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
                       
        def _cal_coor_o3(p0,p1,p3):
            #function to calculate the new point for p3, see document file #2 for detail procedures
            r=f2(p0,p1)/2.*np.tan(np.pi/3)
            norm_vt=p0-p1
            cent_pt=(p0+p1)/2
            a,b,c=norm_vt[0],norm_vt[1],norm_vt[2]
            d=-a*cent_pt[0]-b*cent_pt[1]-c*cent_pt[2]
            u,v,w=p3[0],p3[1],p3[2]
            k=(a*u+b*v+c*w+d)/(a**2+b**2+c**2)
            #projection of O3 to the normal plane see http://www.9math.com/book/projection-point-plane for detail algorithm
            O3_proj=np.array([u-a*k,v-b*k,w-c*k])
            cent_proj_vt=O3_proj-cent_pt
            l=f2(O3_proj,cent_pt)
            ptOnCircle_cent_vt=cent_proj_vt/l*r
            ptOnCircle=ptOnCircle_cent_vt+cent_pt
            return ptOnCircle
 
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        p_O3_old=pt_ct2(domain,p_O3_index,offset[2])*basis
        p_O3=_cal_coor_o3(p_O1,p_O2,p_O3_old)
        
        pyramid_distortion=trigonal_pyramid_distortion_shareface.trigonal_pyramid_distortion_shareface(p0=p_O1,p1=p_O2,p2=p_O3,top_angle=top_angle)
        pyramid_distortion.cal_apex_coor()
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        dif_value=(p_O3-p_O3_old)/basis
        domain.dx1[p_O3_index],domain.dy1[p_O3_index],domain.dz1[p_O3_index]=dif_value[0],dif_value[1],dif_value[2]
        #_add_sorbate(domain=domain,id_sorbate=attach_atm_id_third[0],el='O',sorbate_v=(p_O3-_translate_offset_symbols(offset[2]))/basis)
        
        
        
    def adding_sorbate_pyramid_distortion2(self,domain,edge_offset=[0.,0.],top_angle=1.,top_angle_base=np.pi/3,switch=False,p2_switch=1.,phi=0.,attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1']):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        pyramid_distortion=trigonal_pyramid_distortion2.trigonal_pyramid_distortion(p0=p_O1,p1=p_O2,top_angle=top_angle,top_angle_base=top_angle_base,len_offset=edge_offset,p2_switch=p2_switch)
        pyramid_distortion.all_in_all(switch=switch,phi=phi)
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=pyramid_distortion.p2/basis)

    def adding_sorbate_pyramid_distortion3(self,domain,edge_offset=[0.,0.],top_angle=1.,switch=False,phi1=0.,phi2=0.,attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1']):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        pyramid_distortion=trigonal_pyramid_distortion3.trigonal_pyramid_distortion(p0=p_O1,p1=p_O2,top_angle=top_angle,len_offset=edge_offset)
        pyramid_distortion.all_in_all(switch=switch,phi=phi1,phi2=phi2)
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=pyramid_distortion.p2/basis)
        
    def adding_sorbate_pyramid_distortion4(self,domain,edge_offset=[0.,0.],top_angle=1.,switch=False,phi=0.,attach_atm_ids=['id1','id2'],offset=[None,None],pb_id='pb_id',O_id=['id1']):
        #The added sorbates (including Pb and one Os) will form a edge-distorted trigonal pyramid configuration with the attached ones
        #same way,refer to trigonal_pyramid_distortion4 for detail
        p_O1_index=np.where(domain.id==attach_atm_ids[0])
        p_O2_index=np.where(domain.id==attach_atm_ids[1])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        pyramid_distortion=trigonal_pyramid_distortion4.trigonal_pyramid_distortion(p0=p_O1,p1=p_O2,top_angle=top_angle,len_offset=edge_offset)
        pyramid_distortion.all_in_all(switch=switch,phi=phi)
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=pyramid_distortion.apex/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id[0],el='O',sorbate_v=pyramid_distortion.p2/basis)
        
    def adding_sorbate_hexahedral_distortion_shareface(self,domain,open_angle=1.,theta_top_down=0.,attach_atm_ids_ref=['id1','id2'],attach_atm_id_third=['id3'],offset=[None,None,None],pb_id='pb_id',O_id='O_id'):
        #tridentate configuration under trigonal dipyramid situation
        #see the hexahedra_distortion for detail
        #about calculation of the new third oxygen coors see documents in adding_sorbate_pyramid_distortion_B
        #open_angle must be higher than the top angle of the triangle determined by the three oxygen,
        p_O1_index=np.where(domain.id==attach_atm_ids_ref[0])
        p_O2_index=np.where(domain.id==attach_atm_ids_ref[1])
        p_O3_index=np.where(domain.id==attach_atm_id_third[0])
        basis=np.array([5.038,5.434,7.3707])
        
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-1.,0.,0.])
            elif symbol=='+x':return np.array([1.,0.,0.])
            elif symbol=='-y':return np.array([0.,-1.,0.])
            elif symbol=='+y':return np.array([0.,1.,0.])
            elif symbol=='+x+y':return np.array([1.,1.,0.])
            elif symbol==None:return np.array([0.,0.,0.])

        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        pt_ct=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0]+domain.dx1[p_O1_index][0]+domain.dx2[p_O1_index][0]+domain.dx3[p_O1_index][0]+domain.dx4[p_O1_index][0],\
                       domain.y[p_O1_index][0]+domain.dy1[p_O1_index][0]+domain.dy2[p_O1_index][0]+domain.dy3[p_O1_index][0]+domain.dy4[p_O1_index][0],\
                       domain.z[p_O1_index][0]+domain.dz1[p_O1_index][0]+domain.dz2[p_O1_index][0]+domain.dz3[p_O1_index][0]+domain.dz4[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
        
        pt_ct2=lambda domain,p_O1_index,symbol:np.array([domain.x[p_O1_index][0],\
                       domain.y[p_O1_index][0],\
                       domain.z[p_O1_index][0]])\
                       +_translate_offset_symbols(symbol)
                       
        def _cal_coor_o3(p0,p1,p3):
            r=f2(p0,p1)/2.*np.tan(np.pi/3)
            norm_vt=p0-p1
            cent_pt=(p0+p1)/2
            a,b,c=norm_vt[0],norm_vt[1],norm_vt[2]
            d=-a*cent_pt[0]-b*cent_pt[1]-c*cent_pt[2]
            u,v,w=p3[0],p3[1],p3[2]
            k=(a*u+b*v+c*w+d)/(a**2+b**2+c**2)
            O3_proj=np.array([u-a*k,v-b*k,w-c*k])
            return O3_proj
 
        p_O1=pt_ct(domain,p_O1_index,offset[0])*basis
        p_O2=pt_ct(domain,p_O2_index,offset[1])*basis
        p_O3_old=pt_ct2(domain,p_O3_index,offset[2])*basis
        p_O3=_cal_coor_o3(p_O1,p_O2,p_O3_old)
        #print p_O1,p_O2,p_O3
        hexahedra_distortion_case=hexahedra_distortion.share_face(face=np.array([list(p_O3),list(p_O2),list(p_O1)]),open_angle=open_angle,r_top_down=None,theta_top_down=theta_top_down,switch=True)
        hexahedra_distortion_case.share_face_init(flag='1_2')
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
                
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=hexahedra_distortion_case.center_point/basis)
        _add_sorbate(domain=domain,id_sorbate=O_id,el='O',sorbate_v=hexahedra_distortion_case.p4/basis)
        dif_value=(p_O3-p_O3_old)/basis
        domain.dx1[p_O3_index],domain.dy1[p_O3_index],domain.dz1[p_O3_index]=dif_value[0],dif_value[1],dif_value[2]
        
        #_add_sorbate(domain=domain,id_sorbate=attach_atm_id_third[0],el='O',sorbate_v=(p_O3-_translate_offset_symbols(offset[2]))/basis)
    
    def adding_oxygen(self,domain,o_id,sorbate_coor,r,theta,phi):
        #sorbate_coor and r are in angstrom
        #the sorbate_coor is the origin of a sphere, oxygen added a point determined by r theta and phi
        basis=np.array([5.038,5.434,7.3707])
        x=r*np.cos(phi)*np.sin(theta)
        y=r*np.sin(phi)*np.sin(theta)
        z=r*np.cos(theta)
        o_coor=(np.array([x,y,z])+sorbate_coor)/basis
        o_index=None
        try:
            o_index=np.where(domain.id==o_id)[0][0]
        except:
            domain.add_atom( o_id, "O",  o_coor[0] ,o_coor[1], o_coor[2] ,0.32,     1.00000e+00 ,     1.00000e+00 )
        if o_index!=None:
            domain.x[o_index]=o_coor[0]
            domain.y[o_index]=o_coor[1]
            domain.z[o_index]=o_coor[2]
            
    def updata_sorbate_polyhedra2(self,domain_list=[],id_list=[],polyhedra_index_list=[],offset=0,r=0,theta=0,phi=0):
        #id in id list looks like Os_A_0 or Os_AA_0
        #id1 in domain1(domain1A) has the same setting with id2 in domain2(domain1B)
        #id1 in domain1 correspond to polyhedra list[1]
        #id always indexed from 0,like Os_A_0,the corresponding point in polyhedra can be different
        #depending on shareing mode, like if share edge, id0-->p2 (the p0 and p1 is the shared point), where the offset here is 2
        #after this step the orientation of polyhedra will be having effect on the position of oxygen added
        #the length of all the lists here is 2, representing doing setting equivalently for two chemically equivalent atoms
        #note the id list is like [id_A,id_B],the associated polyhedra index list is [0,2]
        #or [id_AA,id_BB]-->[1,3]
        #if you consider different number of equivalent domain, like 3, then the list should be set accordingly
        for i in range(len(domain_list)):
            id_index=list(domain_list[i].id).index(id_list[i])
            polyhedra_symbol='p'+str(int(id_list[i][-1])+offset)
            p_index=polyhedra_index_list[i]
            new_point_xyz=self.polyhedra_list[p_index].cal_point_in_fit(r=r,theta=theta,phi=phi)
            #print self.polyhedra_list[p_index].T
            #print 'sensor1'+str(r)
            domain_list[i].x[id_index]=new_point_xyz[0]/5.038
            domain_list[i].y[id_index]=new_point_xyz[1]/5.434
            domain_list[i].z[id_index]=new_point_xyz[2]/7.3707
            #print domain_list[i].x[id_index]
    
        
    def updata_sorbate_polyhedra2_batch(self,file):
        f=open(file)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                domain_list,id_list,polyhedra_index_list,offset,r,theta,phi=0,0,0,0,0,0,0
                line_split=line.rsplit(',')
                try:
                    domain_list=[vars(self)[line_split[0]],vars(self)[line_split[1]]]
                    id_list=[line_split[2],line_split[3]]
                    polyhedra_index_list=[int(line_split[4]),int(line_split[5])]
                    offset=int(line_split[6])
                    r=getattr(self.new_var_module,line_split[7])
                    theta=getattr(self.new_var_module,line_split[8])
                    phi=getattr(self.new_var_module,line_split[9])
                except:
                    domain_list=[vars(self)[line_split[0]]]
                    id_list=[line_split[1]]
                    polyhedra_index_list=[int(line_split[2])]
                    offset=int(line_split[3])
                    r=getattr(self.new_var_module,line_split[4])
                    theta=getattr(self.new_var_module,line_split[5])
                    phi=getattr(self.new_var_module,line_split[6])
                self.updata_sorbate_polyhedra2(domain_list,id_list,polyhedra_index_list,offset,r,theta,phi)
        f.close()
        
    def add_sorbates(self,domain,attach_atm_id=[['id1','id2']],el=['Pb'],id=[1],O_id=['_A'],r1=0.1,r2=None,alpha1=1.7,alpha2=None):
        #this function can add multiple sorbates
        #domain is a slab under consideration
        #attach_atm_id is a list of ids to be attached by absorbates,2 by n
        #el is list of element symbol for the first absorbates
        #id is the list of index number to be attached to elment symbol as the id symbol
        #O_id is list, each member will be attached at the end of id of the other absorbates
        #r1 alpha1 associated to the first absorbates, and r2 alpha2 associated to the other absorbates
        #add several lead, and two oxygen attached to each lead atom
        for i in range(len(el)):
            point1_x=domain.x[np.where(domain.id==attach_atm_id[i][0])[0][0]]
            point1_y=domain.y[np.where(domain.id==attach_atm_id[i][0])[0][0]]
            point1_z=domain.z[np.where(domain.id==attach_atm_id[i][0])[0][0]]
            point2_x=domain.x[np.where(domain.id==attach_atm_id[i][1])[0][0]]
            point2_y=domain.y[np.where(domain.id==attach_atm_id[i][1])[0][0]]
            point2_z=domain.z[np.where(domain.id==attach_atm_id[i][1])[0][0]]
            point1=[point1_x,point1_y,point1_z]
            point2=[point2_x,point2_y,point2_z]
            point_sorbate=self._cal_xyz_single(point1,point2,r1,alpha1)
            domain.add_atom(id=el[i]+str(id[i]),element=el[i],x=point_sorbate[0],y=point_sorbate[1],z=point_sorbate[2],u=1.)
            if r2!=None:
                point_sorbate_1,point_sorbate_2=self._cal_xyz_double(point_sorbate,r2,alpha2)
                domain.add_atom(id='Oi_1'+str(O_id[i]),element='O',x=point_sorbate_1[0],y=point_sorbate_1[1],z=point_sorbate_1[2],u=1.)
                domain.add_atom(id='Oi_2'+str(O_id[i]),element='O',x=point_sorbate_2[0],y=point_sorbate_2[1],z=point_sorbate_2[2],u=1.)
        #return domain
    def add_oxygen_pair2(self,domain,ref_id,ref_xy,O_id,v_shift,r,alpha):
    #v_shift and r are in unit of angstrom
        ref_index=np.where(domain.id==ref_id)[0][0]
        basis=np.array([5.038,5.434,7.3707])
        ref_point=[ref_xy[0]*basis[0],ref_xy[1]*basis[1],domain.z[ref_index]*basis[2]+v_shift]
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        point1=np.array([ref_point[0]-x_shift,ref_point[1]-y_shift,ref_point[2]])/basis
        point2=np.array([ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]])/basis
        O_index1=None
        O_index2=None
        try:
            O_index1=np.where(domain.id==O_id[0])[0][0]
            O_index2=np.where(domain.id==O_id[1])[0][0]
        except:
            domain.add_atom( O_id[0], "O",  point1[0] ,point1[1], point1[2] ,0.32,     1.00000e+00 ,     1.00000e+00 )
            domain.add_atom( O_id[1], "O",  point2[0] ,point2[1], point2[2] ,0.32,     1.00000e+00 ,     1.00000e+00 )
        if O_index1!=None:
            domain.x[O_index1],domain.y[O_index1],domain.z[O_index1]=point1[0],point1[1],point1[2]
            domain.x[O_index2],domain.y[O_index2],domain.z[O_index2]=point2[0],point2[1],point2[2]
            
    def add_oxygen_pair(self,domain,O_id,ref_point,r,alpha):
        #add single oxygen pair to a ref_point,which does not stand for an atom, the xyz for this point will be set as
        #three fitting parameters.O_id will be attached at the end of each id for the oxygen
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        point1=ref_point[0]-x_shift,ref_point[1]-y_shift,ref_point[2]
        point2=ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]
        domain.add_atom(id='Os_1'+str(O_id),element='O',x=point1[0],y=point1[1],z=point1[2],u=1.)
        domain.add_atom(id='Os_2'+str(O_id),element='O',x=point2[0],y=point2[1],z=point2[2],u=1.)

    def updata_oxygen_pair(self,domain,ids,ref_point,r,alpha):
        #updata the position information of oxygen pair, to be dropped inside sim func
        #print 'sensor',np.where(domain.id==ids[0]),np.where(domain.id==ids[0])[0]
        index_1=np.where(domain.id==ids[0])[0][0]
        index_2=np.where(domain.id==ids[1])[0][0]
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        domain.x[index_1]=ref_point[0]+x_shift
        domain.y[index_1]=ref_point[1]+y_shift
        domain.z[index_1]=ref_point[2]
        domain.x[index_2]=ref_point[0]-x_shift
        domain.y[index_2]=ref_point[1]-y_shift
        domain.z[index_2]=ref_point[2]
    
    def updata_oxygen_pair_batch(self,filename):
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                domain=vars(self)[line_split[0]]
                ids=[line_split[1],line_split[2]]
                point_x=getattr(self.new_var_module,line_split[3])
                point_y=getattr(self.new_var_module,line_split[4])
                point_z=getattr(self.new_var_module,line_split[5])
                ref_point=[point_x,point_y,point_z-float(line_split[6])]
                r=getattr(self.new_var_module,line_split[7])
                alpha=getattr(self.new_var_module,line_split[8])
                self.updata_oxygen_pair(domain,ids,ref_point,r,alpha)
        f.close()
    
    def add_oxygen_triple_linear(self,domain,O_id,ref_point,r,alpha):
        #add single oxygen pair to a ref_point,which itself stands for an atom, the xyz for this point will be set as
        #three fitting parameters.O_id will be attached at the end of each id for the oxygen
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        point1=ref_point[0]-x_shift,ref_point[1]-y_shift,ref_point[2]
        point2=ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]
        domain.add_atom(id='Os_1'+str(O_id),element='O',x=point1[0],y=point1[1],z=point1[2],u=1.)
        domain.add_atom(id='Os_2'+str(O_id),element='O',x=point2[0],y=point2[1],z=point2[2],u=1.)
        domain.add_atom(id='Os_3'+str(O_id),element='O',x=ref_point[0],y=ref_point[1],z=ref_point[2],u=1.)    
    
    def updata_oxygen_triple_linear(self,domain,ids,ref_point,r,alpha):
        #updata the position information of oxygen pair, to be dropped inside sim func
        #print 'sensor',np.where(domain.id==ids[0]),np.where(domain.id==ids[0])[0]
        index_1=np.where(domain.id==ids[0])[0][0]
        index_2=np.where(domain.id==ids[1])[0][0]
        index_3=np.where(domain.id==ids[2])[0][0]
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        domain.x[index_1]=ref_point[0]+x_shift
        domain.y[index_1]=ref_point[1]+y_shift
        domain.z[index_1]=ref_point[2]
        domain.x[index_2]=ref_point[0]-x_shift
        domain.y[index_2]=ref_point[1]-y_shift
        domain.z[index_2]=ref_point[2]
        domain.x[index_3]=ref_point[0]
        domain.y[index_3]=ref_point[1]
        domain.z[index_3]=ref_point[2]
    
    def updata_oxygen_triple_linear_batch(self,filename):
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                domain=vars(self)[line_split[0]]
                ids=[line_split[1],line_split[2],line_split[3]]
                point_x=getattr(self.new_var_module,line_split[4])
                point_y=getattr(self.new_var_module,line_split[5])
                point_z=getattr(self.new_var_module,line_split[6])
                ref_point=[point_x,point_y,point_z-float(line_split[7])]
                r=getattr(self.new_var_module,line_split[8])
                alpha=getattr(self.new_var_module,line_split[9])
                self.updata_oxygen_triple_linear(domain,ids,ref_point,r,alpha)
        f.close()
    
    def add_oxygen_triple_circle(self,domain,O_id,ref_point,r,alpha1,alpha2,alpha3):
        #add triple oxygen to a ref_point,which itself stands for an atom, the xyz for this point will be set as
        #three fitting parameters.O_id will be attached at the end of each id for the oxygen
        x_shift1=r*np.cos(alpha1)
        y_shift1=r*np.sin(alpha1)
        x_shift2=r*np.cos(alpha2)
        y_shift2=r*np.sin(alpha2)
        x_shift3=r*np.cos(alpha3)
        y_shift3=r*np.sin(alpha3)
        point1=ref_point[0]+x_shift1,ref_point[1]+y_shift1,ref_point[2]
        point2=ref_point[0]+x_shift2,ref_point[1]+y_shift2,ref_point[2]
        point3=ref_point[0]+x_shift3,ref_point[1]+y_shift3,ref_point[2]
        domain.add_atom(id='Os_1'+str(O_id),element='O',x=point1[0],y=point1[1],z=point1[2],u=1.)
        domain.add_atom(id='Os_2'+str(O_id),element='O',x=point2[0],y=point2[1],z=point2[2],u=1.)
        domain.add_atom(id='Os_3'+str(O_id),element='O',x=point3[0],y=point3[1],z=point3[2],u=1.)
        
    def updata_oxygen_triple_circle(self,domain,ids,ref_point,r,alpha1,alpha2,alpha3):
        #updata the position information of oxygen triple, to be dropped inside sim func
        #print 'sensor',np.where(domain.id==ids[0]),np.where(domain.id==ids[0])[0]
        index_1=np.where(domain.id==ids[0])[0][0]
        index_2=np.where(domain.id==ids[1])[0][0]
        index_3=np.where(domain.id==ids[2])[0][0]
        x_shift1=r*np.cos(alpha1)
        y_shift1=r*np.sin(alpha1)
        x_shift2=r*np.cos(alpha2)
        y_shift2=r*np.sin(alpha2)
        x_shift3=r*np.cos(alpha3)
        y_shift3=r*np.sin(alpha3)
        domain.x[index_1]=ref_point[0]+x_shift1
        domain.y[index_1]=ref_point[1]+y_shift1
        domain.z[index_1]=ref_point[2]
        domain.x[index_2]=ref_point[0]+x_shift2
        domain.y[index_2]=ref_point[1]+y_shift2
        domain.z[index_2]=ref_point[2]
        domain.x[index_3]=ref_point[0]+x_shift3
        domain.y[index_3]=ref_point[1]+y_shift3
        domain.z[index_3]=ref_point[2]
        
    def updata_oxygen_triple_circle_batch(self,filename):
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                domain=vars(self)[line_split[0]]
                ids=[line_split[1],line_split[2],line_split[3]]
                point_x=getattr(self.new_var_module,line_split[4])
                point_y=getattr(self.new_var_module,line_split[5])
                point_z=getattr(self.new_var_module,line_split[6])
                ref_point=[point_x,point_y,point_z-float(line_split[7])]
                r=getattr(self.new_var_module,line_split[8])
                alpha1=getattr(self.new_var_module,line_split[9])
                alpha2=getattr(self.new_var_module,line_split[10])
                alpha3=getattr(self.new_var_module,line_split[11])
                self.updata_oxygen_triple_circle(domain,ids,ref_point,r,alpha1,alpha2,alpha3)
        f.close()
        
    def group_sorbates_2(self,domain,attach_atm_id,ids_to_be_attached,r,alpha,beta,gamma):
        #updating the sorbate position, to be dropped inside sim function
        #the same as the group_sorbates except more freedome for the attached sorbates
        #r is the distance between Pb and one of O in this case, alpha is half of the open angle between the sorbates
        #beta is the angle between the normal line and the plane formed by three sorbates
        #gamma is then angle between the x axis and the first edge in the two dimentional space
        #alpha from 0-pi/2, beta from 0-pi/2, gamma from 0-2pi
        index_ref=np.where(domain.id==attach_atm_id)[0][0]
        index_1=np.where(domain.id==ids_to_be_attached[0])[0][0]
        index_2=np.where(domain.id==ids_to_be_attached[1])[0][0]
        ref_x=domain.x[index_ref]+domain.dx1[index_ref]+domain.dx2[index_ref]+domain.dx3[index_ref]
        ref_y=domain.y[index_ref]+domain.dy1[index_ref]+domain.dy2[index_ref]+domain.dy3[index_ref]
        ref_z=domain.z[index_ref]+domain.dz1[index_ref]+domain.dz2[index_ref]+domain.dz3[index_ref]
        z_shift=r*np.cos(alpha)*np.cos(beta)
        #r1 is the edge length of triangle inside the circle, alpha1 is the half open angle of that triangle
        r1=(r**2-z_shift**2)**0.5
        alpha1=np.arcsin(r*np.sin(alpha)/r1)
        point1_x_shift=r1*np.cos(gamma)
        point1_y_shift=r1*np.sin(gamma)
        point2_x_shift=r1*np.cos(gamma+2.*alpha1)
        point2_y_shift=r1*np.sin(gamma+2.*alpha1)
        domain.x[index_1]=ref_x+point1_x_shift
        domain.y[index_1]=ref_y+point1_y_shift
        domain.z[index_1]=ref_z+z_shift
        domain.x[index_2]=ref_x+point2_x_shift
        domain.y[index_2]=ref_y+point2_y_shift
        domain.z[index_2]=ref_z+z_shift
    
    def group_sorbates_2_batch(self,filename):
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.split(',')
                domain=vars(self)[line_split[0]]
                attach_id=line_split[1]
                tobe_attached=line_split[2:4]
                r=getattr(self.new_var_module,line_split[4])
                alpha=getattr(self.new_var_module,line_split[5])
                beta=getattr(self.new_var_module,line_split[6])
                gamma=getattr(self.new_var_module,line_split[7])
                self.group_sorbates_2(domain,attach_id,tobe_attached,r,alpha,beta,gamma)
        f.close()
            
    def group_sorbates(self,domain,attach_atm_id,sorbate_ids,r1,alpha1,z_shift):
        #group the oxygen pair to the absorbate specified,attach_atm_id='Pb1',sorbate_ids=[]
        index_ref=np.where(domain.id==attach_atm_id)[0][0]
        index_1=np.where(domain.id==sorbate_ids[0])[0][0]
        index_2=np.where(domain.id==sorbate_ids[1])[0][0]
        ref_x=domain.x[index_ref]+domain.dx1[index_ref]+domain.dx2[index_ref]+domain.dx3[index_ref]
        ref_y=domain.y[index_ref]+domain.dy1[index_ref]+domain.dy2[index_ref]+domain.dy3[index_ref]
        ref_z=domain.z[index_ref]+domain.dz1[index_ref]+domain.dz2[index_ref]+domain.dz3[index_ref]
        O1_point,O2_point=self._cal_xyz_double(ref_point=[ref_x,ref_y,ref_z],r=r1,alpha=alpha1,z_shift=z_shift)
        domain.x[index_1],domain.y[index_1],domain.z[index_1]=O1_point[0],O1_point[1],O1_point[2]
        domain.x[index_2],domain.y[index_2],domain.z[index_2]=O2_point[0],O2_point[1],O2_point[2]
        
    def updata_sorbates(self,domain,id1,r1,alpha1,z_shift,attach_atm_id=['id1','id2'],id2=[],r2=None,alpha2=None):
        #old version of updating,less freedome for Pb sorbates
        #group all sorbates to the first layer oxygen pair
        #domain is a slab under consideration
        #id1 is the id for the first absorbate(Pb), r1 is positive value, alpha1 is angle lower than pi
        #attach_atm_id is a list of ids of first atoms(oxy)
        #id2 is a list of two pair absorbates, r2 is positive value, alpha2 is anlge less than pi
        index_1=np.where(domain.id==attach_atm_id[0])[0][0]
        index_2=np.where(domain.id==attach_atm_id[1])[0][0]
        point1_x=domain.x[index_1]+domain.dx1[index_1]+domain.dx2[index_1]+domain.dx3[index_1]
        point1_y=domain.y[index_1]+domain.dy1[index_1]+domain.dy2[index_1]+domain.dy3[index_1]
        point1_z=domain.z[index_1]+domain.dz1[index_1]+domain.dz2[index_1]+domain.dz3[index_1]
        point2_x=domain.x[index_2]+domain.dx1[index_2]+domain.dx2[index_2]+domain.dx3[index_2]
        point2_y=domain.y[index_2]+domain.dy1[index_2]+domain.dy2[index_2]+domain.dy3[index_2]
        point2_z=domain.z[index_2]+domain.dz1[index_2]+domain.dz2[index_2]+domain.dz3[index_2]
        
        point1=[point1_x,point1_y,point1_z]
        point2=[point2_x,point2_y,point2_z]
        point_sorbate=self._cal_xyz_single(point1,point2,r1,alpha1)
        domain.x[np.where(domain.id==id1)[0][0]]=point_sorbate[0]
        domain.y[np.where(domain.id==id1)[0][0]]=point_sorbate[1]
        domain.z[np.where(domain.id==id1)[0][0]]=point_sorbate[2]
        
        if r2!=None:
            point_sorbate_1,point_sorbate_2=self._cal_xyz_double(point_sorbate,r2,alpha2,z_shift)
            
            domain.x[np.where(domain.id==id2[0])[0][0]]=point_sorbate_1[0]
            domain.y[np.where(domain.id==id2[0])[0][0]]=point_sorbate_1[1]
            domain.z[np.where(domain.id==id2[0])[0][0]]=point_sorbate_1[2]
            
            domain.x[np.where(domain.id==id2[1])[0][0]]=point_sorbate_2[0]
            domain.y[np.where(domain.id==id2[1])[0][0]]=point_sorbate_2[1]
            domain.z[np.where(domain.id==id2[1])[0][0]]=point_sorbate_2[2]
        #return domain
    
    def _cal_xyz_single(self,point1,point2,r,alpha):
        #point1=[x1,y1,z1],point2=[x2,y2,z2],r is a value, alpha is angle less than pi
        slope_pt1_pt2=(point1[1]-point2[1])/(point1[0]-point2[0])
        slope_new1=-1./slope_pt1_pt2
        cent_point=[(point1[0]+point2[0])/2.,(point1[1]+point2[1])/2.]
        dist_pt12=((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)**0.5
        tan_theta=r*np.cos(alpha)/(dist_pt12/2.)
        slope_new2=(slope_pt1_pt2+tan_theta)/(1.-slope_pt1_pt2*tan_theta)
        #slope_new1 and cent_point form a line equation
        #slope_new2 and point2 form another line equation
        A=np.array([[-slope_new1,1.],[-slope_new2,1.]])
        C=np.array([cent_point[1]-slope_new1*cent_point[0],point2[1]-slope_new2*point2[0]])
        xy=np.dot(inv(A),C)
        return [xy[0],xy[1],point1[2]+r*np.sin(alpha)]
        
    def _cal_xyz_double(self,ref_point,r,alpha,z_shift=0.1):
    #ref_point=[x1,y1,z1],r is a positive value, alpha an angle less than pi, z_shift is positive value represent shift at z direction
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        new_point1=[ref_point[0]+x_shift,ref_point[1]+y_shift,ref_point[2]+z_shift]
        new_point2=[2.*ref_point[0]-new_point1[0],2.*ref_point[1]-new_point1[1],ref_point[2]+z_shift]
        return new_point1,new_point2
    
    def grouping_sequence_layer(self, domain=[], first_atom_id=[],sym_file={},id_match_in_sym={},layers_N=1,use_sym=False):
        #group the atoms at the same layer in one domain and the associated atoms in its chemically equivalent domain
        #so 4 atoms will group together if consider two chemical equivalent domain
        #domain is list of two chemical equivalent domains
        #first_atom_id is list of first id in id array of two domains
        #sym_file is a library of symmetry file names, the keys are element symbols
        #id_match_in_sym is a library of ids, the order of which match the symmetry operation in the associated sym file
        #layers_N is the number of layer you consider for grouping operation
        #use_sym is a flag to choose the shifting rule (symmetry basis or not)
        atm_gp_list=[]
        for i in range(layers_N):
            index_1=np.where(domain[0].id==first_atom_id[0])[0][0]+i*2
            temp_atm_gp=model.AtomGroup(slab=domain[0],id=str(domain[0].id[index_1]),id_in_sym_file=id_match_in_sym[str(domain[0].el[index_1])],use_sym=use_sym,filename=sym_file[str(domain[0].el[index_1])])
            temp_atm_gp.add_atom(domain[0],str(domain[0].id[index_1+1]))
            index_2=np.where(domain[1].id==first_atom_id[1])[0][0]+i*2
            temp_atm_gp.add_atom(domain[1],str(domain[1].id[index_2]))
            temp_atm_gp.add_atom(domain[1],str(domain[1].id[index_2+1]))
            atm_gp_list.append(temp_atm_gp)

        return atm_gp_list
        
    def grouping_sequence_layer2(self, domain=[], first_atom_id=[],sym_file={},id_match_in_sym={},layers_N=1,use_sym=False):
        #different from first edition, we consider only one domain
        #group the atoms at the same layer in one domain and the associated atoms in its chemically equivalent domain
        #so 4 atoms will group together if consider two chemical equivalent domain
        #domain is list of two chemical equivalent domains
        #first_atom_id is list of first id in id array of two domains
        #sym_file is a library of symmetry file names, the keys are element symbols
        #id_match_in_sym is a library of ids, the order of which match the symmetry operation in the associated sym file
        #layers_N is the number of layer you consider for grouping operation
        #use_sym is a flag to choose the shifting rule (symmetry basis or not)
        atm_gp_list=[]
        for i in range(layers_N):
            index_1=np.where(domain[0].id==first_atom_id[0])[0][0]+i*2
            temp_atm_gp=model.AtomGroup(slab=domain[0],id=str(domain[0].id[index_1]),id_in_sym_file=id_match_in_sym[str(domain[0].el[index_1])],use_sym=use_sym,filename=sym_file[str(domain[0].el[index_1])])
            temp_atm_gp.add_atom(domain[0],str(domain[0].id[index_1+1]))
            #index_2=np.where(domain[1].id==first_atom_id[1])[0][0]+i*2
            #temp_atm_gp.add_atom(domain[1],str(domain[1].id[index_2]))
            #temp_atm_gp.add_atom(domain[1],str(domain[1].id[index_2+1]))
            atm_gp_list.append(temp_atm_gp)

        return atm_gp_list
    def grouping_discrete_layer(self,domain=[],atom_ids=[],sym_file=None,id_match_in_sym=[],use_sym=False):
        #we usually do discrete grouping for sorbates, so there is no symmetry used in this case
        atm_gp=model.AtomGroup(id_in_sym_file=id_match_in_sym,filename=sym_file,use_sym=use_sym)
        for i in range(len(domain)):
            atm_gp.add_atom(domain[i],atom_ids[i])
        return atm_gp
        
    def grouping_discrete_layer_batch(self,filename):
        gp_list=[]
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                N_half=len(line_split)/2
                domains=[]
                ids=[]
                for i in range(N_half):
                    domains.append(vars(self)[line_split[i]])
                    ids.append(line_split[i+N_half])
                gp_list.append(self.grouping_discrete_layer(domain=domains,atom_ids=ids))
        f.close()
        return tuple(gp_list)
        
    def _extract_list(self,ref_list,extract_index):
        output_list=[]
        for i in extract_index:
            output_list.append(ref_list[i])
        return output_list
        
    def split_number(self,N_str):
        N_list=[]
        for i in range(len(N_str)):
            N_list.append(int(N_str[i]))
        return N_list
        
    def scale_opt(self,atm_gp_list,scale_factor,sign_values=None,flag='u',ref_v=1.):
        #scale the parameter from first layer atom to deeper layer atom
        #dx,dy,dz,u will decrease inward, oc decrease outward usually
        #and note the ref_v for oc and u is the value for inner most atom, while ref_v for the other parameters are values for outer most atoms
        #atm_gp_list is a list of atom group to consider the scaling operation
        #scale_factor is list of values of scale factor, note accummulated product will be used for scaling
        #flag is the parameter symbol
        #ref_v is the reference value to start off 
        if sign_values==None:
            for i in range(len(atm_gp_list)):
                atm_gp_list[i]._set_func(flag)(ref_v*reduce(mul,scale_factor[:i+1]))
        else:
            for i in range(len(atm_gp_list)):
                atm_gp_list[i]._set_func(flag)(ref_v*sign_values[i]*reduce(mul,scale_factor[:i+1]))
    
    def scale_opt_batch(self,filename):
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                atm_gp_list=vars(self)[line_split[0]]
                index_list=self.split_number(line_split[1])
                scale_factor=vars(self)[line_split[2]]
                sign_values=0.
                if line_split[3]=='None':
                    sign_values=None
                else:
                    sign_values=vars(self)[line_split[3]]
                flag=line_split[4]
                ref_v=0.
                try:
                    ref_v=float(line_split[5])
                except:
                    ref_v=vars(self)[line_split[5]]
                self.scale_opt(self._extract_list(atm_gp_list,index_list),scale_factor,sign_values,flag,ref_v)
        f.close()
        
    def scale_opt2(self,atm_gp_list,scale_factor,sign_values=None,flag='u',ref_v=1.):
        #scale the parameter from first layer atom to deeper layer atom
        #dx,dy,dz,u will decrease inward, oc decrease outward usually
        #and note the ref_v for oc and u is the value for inner most atom, while ref_v for the other parameters are values for outer most atoms
        #atm_gp_list is a list of atom group to consider the scaling operation
        #scale_factor is list of values of scale factor, note accummulated product will be used for scaling
        #flag is the parameter symbol
        #ref_v is the reference value to start off 
        if sign_values==None:
            if (flag=='dx'):
            #considering inplane movement for dxdy, the movement (distance from the starting position) is scaled
            #the dxdy is finally determined by the movement magnitute and the rotation angle at each layer
                for i in range(len(atm_gp_list)):
                    atm_gp_list[i]._set_func(flag)(ref_v*reduce(mul,scale_factor[:i+1])*np.cos(vars(self)['scale_values_all_inp_angle'][i]))
            
            elif flag=='dy':
                for i in range(len(atm_gp_list)):
                    atm_gp_list[i]._set_func(flag)(ref_v*reduce(mul,scale_factor[:i+1])*np.sin(vars(self)['scale_values_all_inp_angle'][i]))
            else:
                for i in range(len(atm_gp_list)):
                    atm_gp_list[i]._set_func(flag)(ref_v*reduce(mul,scale_factor[:i+1]))
        else:
            for i in range(len(atm_gp_list)):
                atm_gp_list[i]._set_func(flag)(ref_v*sign_values[i]*reduce(mul,scale_factor[:i+1]))
    
    def scale_opt_batch2(self,filename):
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                atm_gp_list=vars(self)[line_split[0]]
                index_list=self.split_number(line_split[1])
                scale_factor=vars(self)[line_split[2]]
                sign_values=0.
                if line_split[3]=='None':
                    sign_values=None
                else:
                    sign_values=vars(self)[line_split[3]]
                flag=line_split[4]
                ref_v=0.
                try:
                    ref_v=float(line_split[5])
                except:
                    ref_v=vars(self)[line_split[5]]
                
                self.scale_opt2(self._extract_list(atm_gp_list,index_list),scale_factor,sign_values,flag,ref_v)
        f.close()
        
    def scale_opt3(self,atm_gp_list,scale_factor,sign_values=None,flag='u',ref_v=1.):
        #scale the parameter from first layer atom to deeper layer atom
        #dx,dy,dz,u will decrease inward, oc decrease outward usually
        #and note the ref_v for oc and u is the value for inner most atom, while ref_v for the other parameters are values for outer most atoms
        #atm_gp_list is a list of atom group to consider the scaling operation
        #scale_factor is list of values of scale factor, note accummulated product will be used for scaling
        #flag is the parameter symbol
        #ref_v is the reference value to start off 
        if sign_values==None:
            f2=lambda domain,id_name:np.where(domain.id==id_name)[0][0]
            if (flag=='dx'):
            #considering inplane movement for dxdy, the movement (distance from the starting position) is scaled
            #the dxdy is finally determined by the movement magnitute and the rotation angle at each layer
                for i in range(len(atm_gp_list)):
                    atm_gp_list[i].slabs[0].dx1[f2(atm_gp_list[i].slabs[0],atm_gp_list[i].ids[0])]=ref_v*reduce(mul,scale_factor[:i+1])*np.cos(vars(self)['scale_values_all_inp_angle'][i*2])
                    atm_gp_list[i].slabs[1].dx1[f2(atm_gp_list[i].slabs[1],atm_gp_list[i].ids[1])]=ref_v*reduce(mul,scale_factor[:i+1])*np.cos(vars(self)['scale_values_all_inp_angle'][i*2+1])

            elif flag=='dy':
                for i in range(len(atm_gp_list)):
                    atm_gp_list[i].slabs[0].dy1[f2(atm_gp_list[i].slabs[0],atm_gp_list[i].ids[0])]=ref_v*reduce(mul,scale_factor[:i+1])*np.cos(vars(self)['scale_values_all_inp_angle'][i*2])
                    atm_gp_list[i].slabs[1].dy1[f2(atm_gp_list[i].slabs[1],atm_gp_list[i].ids[1])]=ref_v*reduce(mul,scale_factor[:i+1])*np.cos(vars(self)['scale_values_all_inp_angle'][i*2+1])
            else:
                for i in range(len(atm_gp_list)):
                    atm_gp_list[i]._set_func(flag)(ref_v*reduce(mul,scale_factor[:i+1]))
        else:
            for i in range(len(atm_gp_list)):
                atm_gp_list[i]._set_func(flag)(ref_v*sign_values[i]*reduce(mul,scale_factor[:i+1]))
    
    def scale_opt_batch3(self,filename):
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                atm_gp_list=vars(self)[line_split[0]]
                index_list=self.split_number(line_split[1])
                scale_factor=vars(self)[line_split[2]]
                sign_values=0.
                if line_split[3]=='None':
                    sign_values=None
                else:
                    sign_values=vars(self)[line_split[3]]
                flag=line_split[4]
                ref_v=0.
                try:
                    ref_v=float(line_split[5])
                except:
                    ref_v=vars(self)[line_split[5]]
                
                self.scale_opt3(self._extract_list(atm_gp_list,index_list),scale_factor,sign_values,flag,ref_v)
        f.close()

    def set_new_vars(self,head_list=['u_Fe_'],N_list=[2]):
    #set new vars 
    #head_list is a list of heading test for a new variable,N_list is the associated number of each set of new variable to be created
        for head,N in zip(head_list,N_list):
            for i in range(N):
                getattr(self.new_var_module,'new_var')(head+str(i+1),1.)
    
    def set_discrete_new_vars_batch(self,filename):
    #set discrete new vars
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                #print line_split
                getattr(self.new_var_module,'new_var')(line_split[0],float(line_split[1]))
        f.close()
    
    def norm_sign(self,value,scale=1.):
        if value<=0.5:
            return -scale
        elif value>0.5:
            return scale
            
    def init_sim_batch(self,filename):
        f=open(filename)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                if (line_split[0]=='ocu')|(line_split[0]=='scale'):
                    tmp_list=[]
                    for i in range(len(line_split)-3):
                        tmp_list.append(getattr(self.new_var_module,line_split[i+2]))
                    setattr(self,line_split[1],tmp_list)
                elif line_split[0]=='ref':
                    tmp=getattr(vars(self)[line_split[2]],line_split[3])()
                    setattr(self,line_split[1],tmp)
                elif line_split[0]=='ref_new':
                    tmp=getattr(self.new_var_module,line_split[2])
                    setattr(self,line_split[1],tmp)
                elif line_split[0]=='sign':
                    tmp_list=[]
                    for i in range(len(line_split)-3):
                        tmp_list.append(self.norm_sign(getattr(self.new_var_module,line_split[i+2])))
                    setattr(self,line_split[1],tmp_list)
                    
    def adding_oxygen_pair_sphere(self,domain,o_id_list=[],sorbate_id='O_1',r=1.,theta_list=[],phi_list=[]):
        #sorbate_coor and r are in angstrom
        #the sorbate_coor is the origin of a sphere, oxygen added at point determined by r theta and phi
        #two oxygens have the same r value
        basis=np.array([5.038,5.434,7.3707])
        index_1=np.where(domain.id==sorbate_id)[0][0]
        ref_x=domain.x[index_1]+domain.dx1[index_1]+domain.dx2[index_1]+domain.dx3[index_1]
        ref_y=domain.y[index_1]+domain.dy1[index_1]+domain.dy2[index_1]+domain.dy3[index_1]
        ref_z=domain.z[index_1]+domain.dz1[index_1]+domain.dz2[index_1]+domain.dz3[index_1]
        sorbate_coor=np.array([ref_x,ref_y,ref_z])*basis
        x1,x2=r*np.cos(phi_list[0])*np.sin(theta_list[0]),r*np.cos(phi_list[1])*np.sin(theta_list[1])
        y1,y2=r*np.sin(phi_list[0])*np.sin(theta_list[0]),r*np.sin(phi_list[1])*np.sin(theta_list[1])
        z1,z2=r*np.cos(theta_list[0]),r*np.cos(theta_list[1])
        o1_coor=(np.array([x1,y1,z1])+sorbate_coor)/basis
        o2_coor=(np.array([x2,y2,z2])+sorbate_coor)/basis
        o1_index=None
        o2_index=None
        try:
            o1_index=np.where(domain.id==o_id_list[0])[0][0]
            o2_index=np.where(domain.id==o_id_list[1])[0][0]
        except:
            domain.add_atom( o_id_list[0], "O",  o1_coor[0] ,o1_coor[1], o1_coor[2] ,0.32,     1.00000e+00 ,     1.00000e+00 )
            domain.add_atom( o_id_list[1], "O",  o2_coor[0] ,o2_coor[1], o2_coor[2] ,0.32,     1.00000e+00 ,     1.00000e+00 )
        if o1_index!=None:
            domain.x[o1_index]=o1_coor[0]
            domain.y[o1_index]=o1_coor[1]
            domain.z[o1_index]=o1_coor[2]

            domain.x[o2_index]=o2_coor[0]
            domain.y[o2_index]=o2_coor[1]
            domain.z[o2_index]=o2_coor[2]
                    
    def cal_bond_valence(self,domain,filename,print_file=False,tag=''):
    #calculate the bond valence for an atom with specified surroundring atoms by using the equation s=exp[(r0-r)/B]
        f=open(filename)
        lines=f.readlines()
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        def _get_coor(domain,index,label):
            coor=0.
            if label=='+x':
                coor=np.array([domain.x[index]+domain.dx1[index]+1.,domain.y[index]+domain.dy1[index],domain.z[index]+domain.dz1[index]])*basis
            elif label=='-x':
                coor=np.array([domain.x[index]+domain.dx1[index]-1.,domain.y[index]+domain.dy1[index],domain.z[index]+domain.dz1[index]])*basis
            elif label=='+y':
                coor=np.array([domain.x[index]+domain.dx1[index],domain.y[index]+domain.dy1[index]+1.,domain.z[index]+domain.dz1[index]])*basis
            elif label=='-y':
                coor=np.array([domain.x[index]+domain.dx1[index],domain.y[index]+domain.dy1[index]-1.,domain.z[index]+domain.dz1[index]])*basis
            else:
                coor=np.array([domain.x[index]+domain.dx1[index],domain.y[index]+domain.dy1[index],domain.z[index]+domain.dz1[index]])*basis
            return coor
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                index_list=[]
                r0=float(line_split[-2])
                for i in range((len(line_split)-1)/2):
                    index=np.where(domain.id==line_split[i*2])[0][0]
                    index_list.append([index,line_split[i*2+1]])
                bond_valence=0.
                for index in index_list[1:]:
                    bond_len=f2(_get_coor(domain,index_list[0][0],index_list[0][1]),_get_coor(domain,index[0],index[1]))
                    bond_valence=bond_valence+np.exp((r0-bond_len)/0.37)
                bond_valence_container[line_split[0]]=bond_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+tag+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
                f.close()
        return bond_valence_container
        
    def cal_bond_valence2(self,domain,center_atom_id,searching_range=3.,print_file=False):
    #calculate the bond valence for an atom with specified surroundring atoms by using the equation s=exp[(r0-r)/B]
    #the atoms within the searching range (in unit of A) will be counted for the calculation of bond valence
        bond_valence_container={}
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index],domain.y[index]+domain.dy1[index],domain.z[index]+domain.dz1[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        index=np.where(domain.id==center_atom_id)[0][0]
        for i in range(len(domain.id)):
            if (f2(f1(domain,index),f1(domain,i))<=searching_range)&(f2(f1(domain,index),f1(domain,i))!=0.):
                r0=0
                if ((domain.el[index]=='Pb')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Pb')):r0=2.112
                elif ((domain.el[index]=='Fe')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Fe')):r0=1.759
                elif ((domain.el[index]=='Sb')&(domain.el[i]=='O'))|((domain.el[index]=='O')&(domain.el[i]=='Sb')):r0=1.973
                else:r0=-10
                bond_valence_container[domain.id[i]]=np.exp((r0-f2(f1(domain,index),f1(domain,i)))/0.37)
        sum_valence=0.
        for key in bond_valence_container.keys():
            sum_valence=sum_valence+bond_valence_container[key]
        bond_valence_container['total_valence']=sum_valence
        if print_file==True:
            f=open('/home/tlab/sphalerite/jackey/model2/files_pb/bond_valence_'+center_atom_id+'.txt','w')
            for i in bond_valence_container.keys():
                s = '%-5s   %7.5e\n' % (i,bond_valence_container[i])
                f.write(s)
            f.close()
        return bond_valence_container
        
    def cal_top_oxygen(self,domain,Fe_id,O_id_z,O_id_xy,r,theta,phi,offset=[None,None,None,None]):
    #to calculate the top oxygen position in fragtion,the Fe_id specified Fe atom will be set as the origin of the spherical coordinate system
    #O_z-Fe is the vector of z direction, while this vector and the Fe-O_xy[0],Fe-O_xy[1] form a base set, the normalized orthogonal base set will be
    #computed as the new coordinate frame, then the O top position will be first defined in the spherical frame by specifying r, theta and phi pars
    #then transform to the original coordinate system
    #offset correspond to Fe, O_z, O_xy[0] and O_xy[1], respectively.
        basis=np.array([5.038,5.434,7.3707])
        x0_v,y0_v,z0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])
        f1=lambda domain,id:basis*np.array([domain.x[np.where(domain.id==id)[0][0]],domain.y[np.where(domain.id==id)[0][0]],domain.z[np.where(domain.id==id)[0][0]]])
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        f3=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
        def _translate_offset_symbols(symbol):
            if symbol=='-x':return np.array([-5.038,0.,0.])
            elif symbol=='+x':return np.array([5.038,0.,0.])
            elif symbol=='-y':return np.array([0.,-5.434,0.])
            elif symbol=='+y':return np.array([0.,5.434,0.])
            elif symbol==None:return np.array([0.,0.,0.])
            
        x1,x2,x3=(f1(domain,Fe_id)+_translate_offset_symbols(offset[0]))-(f1(domain,O_id_z)+_translate_offset_symbols(offset[1])),\
                        (f1(domain,O_id_xy[0])+_translate_offset_symbols(offset[2]))-(f1(domain,O_id_z)+_translate_offset_symbols(offset[1])),\
                        (f1(domain,O_id_xy[1])+_translate_offset_symbols(offset[3]))-(f1(domain,O_id_z)+_translate_offset_symbols(offset[1]))
        v1=x1
        v2=x2-(np.dot(x2,v1)/np.dot(v1,v1))*v1
        v3=x3-np.dot(x3,v1)/np.dot(v1,v1)*v1-np.dot(x3,v2)/np.dot(v2,v2)*v2
        v_z=v1/f2(v1,np.array([0.,0.,0.]))
        v_x=v2/f2(v2,np.array([0.,0.,0.]))
        v_y=v3/f2(v3,np.array([0.,0.,0.]))
        T=f3(x0_v,y0_v,z0_v,v_x,v_y,v_z)
        x_new=r*np.cos(phi)*np.sin(theta)
        y_new=r*np.sin(phi)*np.sin(theta)
        z_new=r*np.cos(theta)
        pt_original_fraction=(np.dot(np.linalg.inv(T),np.array([x_new,y_new,z_new]))+f1(domain,Fe_id))/basis
        return pt_original_fraction
    
    def scale_in_symmetry(self,domain,center_id,scaler,ref_lib,off_set,center_offset=None):
        #THE atom will only be allowed to move along the bond vector,to do that we need a center point defined by
        #center_id, and a reference point defining the other end of the vector which is specified by ref_lib
        #you can group several atoms together in the reference library, they will have the same scaler
        #off_set is defined to account the arbitrary movement along x or y directioin for the ref_lib
        #this function must be in sim and after the sorbate updating function, otherwise error will be seen.
        def _offset_translator(offset):
            if offset=='+x':
                return np.array([1.,0.,0.])
            elif offset=='-x':
                return np.array([-1.,0.,0.])
            elif offset=='+y':
                return np.array([0.,1.,0.])
            elif offset=='-y':
                return np.array([0.,-1.,0.])
            else:
                return np.array([0.,0.,0.]) 
        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
        center_coor=_extract_coor(domain,center_id)+_offset_translator(center_offset)
        
        for i in ref_lib.keys():
            bond_vt=ref_lib[i]-center_coor
            bond_vt_scaled=bond_vt*scaler
            ref_coor_scaled=bond_vt_scaled+center_coor
            offset=_offset_translator(off_set[i])
            domain.x[np.where(domain.id==i)[0][0]]=ref_coor_scaled[0]
            domain.y[np.where(domain.id==i)[0][0]]=ref_coor_scaled[1]
            domain.z[np.where(domain.id==i)[0][0]]=ref_coor_scaled[2]

            domain.dx2[np.where(domain.id==i)[0][0]]=-offset[0]
            domain.dy2[np.where(domain.id==i)[0][0]]=-offset[1]
            domain.dz2[np.where(domain.id==i)[0][0]]=-offset[2]
            
    def scale_in_symmetry3(self,domain,center_id,scaler,ref_lib,center_offset=None):
        #difference is no off_set of the surrounding atoms, but there is center_offset
        #THE atom will only be allowed to move along the bond vector,to do that we need a center point defined by
        #center_id, and a reference point defining the other end of the vector which is specified by ref_lib
        #you can group several atoms together in the reference library, they will have the same scaler
        #off_set is defined to account the arbitrary movement along x or y directioin for the ref_lib
        #this function must be in sim and after the sorbate updating function, otherwise error will be seen.
        def _offset_translator(offset):
            if offset=='+x':
                return np.array([1.,0.,0.])
            elif offset=='-x':
                return np.array([-1.,0.,0.])
            elif offset=='+y':
                return np.array([0.,1.,0.])
            elif offset=='-y':
                return np.array([0.,-1.,0.])
            else:
                return np.array([0.,0.,0.]) 
        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
        center_coor=_extract_coor(domain,center_id)+_offset_translator(center_offset)
        
        for i in ref_lib.keys():
            bond_vt=ref_lib[i]-center_coor
            bond_vt_scaled=bond_vt*scaler
            ref_coor_scaled=bond_vt_scaled+center_coor
            
            domain.x[np.where(domain.id==i)[0][0]]=ref_coor_scaled[0]
            domain.y[np.where(domain.id==i)[0][0]]=ref_coor_scaled[1]
            domain.z[np.where(domain.id==i)[0][0]]=ref_coor_scaled[2]
         
        
    def scale_in_symmetry2(self,domain,center_id,scaler,ref_lib,phi_lib,theta_lib,off_set,center_offset=None):
        #different from the previous one is the oxygen ascaled=bond_vt*scaler
            ref_coor_scaled=bond_vt_scaled+center_coor
            offset=_offset_translator(off_set[i])
            domain.x[np.where(domain.id==i)[0][0]]=ref_coor_scaled[0]
            domain.y[np.where(domain.id==i)[0][0]]=ref_coor_scaled[1]
            domain.z[np.where(domain.id==i)[0][0]]=ref_coor_scaled[2]

            domain.dx2[np.where(domain.id==i)[0][0]]=-offset[0]
            domain.dy2[np.where(domain.id==i)[0][0]]=-offset[1]
            domain.dz2[np.where(domain.id==i)[0][0]]=-offset[2]
            
    def scale_in_symmetry3(self,domain,center_id,scaler,ref_lib,center_offset=None):
        #difference is no off_set of the surrounding atoms, but there is center_offset
        #THE atom will only be allowed to move along the bond vector,to do that we need a center point defined by
        #center_id, and a reference point defining the other end of the vector which is specified by ref_lib
        #you can group several atoms together in the reference library, they will have the same scaler
        #off_set is defined to account the arbitrary movement along x or y directioin for the ref_lib
        #this function must be in sim and after the sorbate updating function, otherwise error will be seen.
        def _offset_translator(offset):
            if offset=='+x':
                return np.array([1.,0.,0.])
            elif offset=='-x':
                return np.array([-1.,0.,0.])
            elif offset=='+y':
                return np.array([0.,1.,0.])
            elif offset=='-y':
                return np.array([0.,-1.,0.])
            else:
                return np.array([0.,0.,0.]) 
        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
        center_coor=_extract_coor(domain,center_id)+_offset_translator(center_offset)
        
        for i in ref_lib.keys():
            bond_vt=ref_lib[i]-center_coor
            bond_vt_scaled=bond_vt*scaler
            ref_coor_scaled=bond_vt_scaled+center_coor
            
            domain.x[np.where(domain.id==i)[0][0]]=ref_coor_scaled[0]
            domain.y[np.where(domain.id==i)[0][0]]=ref_coor_scaled[1]
            domain.z[np.where(domain.id==i)[0][0]]=ref_coor_scaled[2]
         
        
    def scale_in_symmetry2(self,domain,center_id,scaler,ref_lib,phi_lib,theta_lib,off_set,center_offset=None):
        #different from the previous one is the oxygen atoms will not only relax along the bond valence vector,but also will
        #rotate over the vector in some angle defind by theta(a very small range adjacent to 0) and phi (0-2pi)
        #THE atom will only be allowed to move along the bond vector,to do that we need a center point defined by
        #center_id, and a reference point defining the other end of the vector which is specified by ref_lib
        #you can group several atoms together in the reference library, they will have the same scaler
        #off_set is defined to account the arbitrary movement along x or y directioin for the ref_lib
        #this function must be in sim and after the sorbate updating function, otherwise error will be seen.
        
        #anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
        #x2y2z2 are basis of new coor defined in the original frame,new=T.orig
        f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
        a0_v,b0_v,c0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.])                              
        def _offset_translator(offset):
            if offset=='+x':
                return np.array([1.,0.,0.])
            elif offset=='-x':
                return np.array([-1.,0.,0.])
            elif offset=='+y':
                return np.array([0.,1.,0.])
            elif offset=='-y':
                return np.array([0.,-1.,0.])
            else:
                return np.array([0.,0.,0.]) 
        def _extract_coor(domain,id):
            index=np.where(domain.id==id)[0][0]
            x=domain.x[index]+domain.dx1[index]+domain.dx2[index]+domain.dx3[index]
            y=domain.y[index]+domain.dy1[index]+domain.dy2[index]+domain.dy3[index]
            z=domain.z[index]+domain.dz1[index]+domain.dz2[index]+domain.dz3[index]
            return np.array([x,y,z])
        center_coor=_extract_coor(domain,center_id)+_offset_translator(center_offset)
        
        for i in ref_lib.keys():
            bond_vt=ref_lib[i]-center_coor
            bond_vt_scaled=bond_vt*scaler
            c_v=bond_vt/(np.dot(bond_vt,bond_vt)**0.5)
            a_v_i=np.array([1.,1.,((center_coor[0]-1.)*c_v[0]+(center_coor[1]-1.)*c_v[1])/c_v[2]+center_coor[2]])
            a_v=a_v_i/(np.dot(a_v_i,a_v_i)**0.5)
            b_v=np.cross(c_v,a_v)
            T=f1(a0_v,b0_v,c0_v,a_v,b_v,c_v)
            r=np.dot(bond_vt_scaled,bond_vt_scaled)**0.5
            theta=theta_lib[i]
            phi=phi_lib[i]
            ox_ps_new=np.array([r*np.cos(phi)*np.sin(theta),r*np.sin(phi)*np.sin(theta),r*np.cos(theta)])
            ox_ps_org=np.dot(inv(T),ox_ps_new)+center_coor
            ref_coor_scaled=ox_ps_org
            
            offset=_offset_translator(off_set[i])
            domain.x[np.where(domain.id==i)[0][0]]=ref_coor_scaled[0]
            domain.y[np.where(domain.id==i)[0][0]]=ref_coor_scaled[1]
            domain.z[np.where(domain.id==i)[0][0]]=ref_coor_scaled[2]

            domain.dx2[np.where(domain.id==i)[0][0]]=-offset[0]
            domain.dy2[np.where(domain.id==i)[0][0]]=-offset[1]
            domain.dz2[np.where(domain.id==i)[0][0]]=-offset[2]
        
    def outer_sphere_complex(self,domain,cent_point=[0.5,0.5,1.],r0=1.,r1=1.,phi=0.,pb_id='pb1',O_ids=['Os1','Os2','Os3']):
        #add a regular trigonal pyramid motiff above the surface representing the outer sphere complexation
        #the pyramid is oriented either Oxygen base top (when r1 is negative) or apex top (when r1 is positive)
        #cent_point in frational coordinate is the center point of the based triangle
        #r0 in ansgtrom is the distance between cent_point and oxygens in the based
        #r1 in angstrom is the distance bw cent_point and apex
        a,b,c=5.038,5.434,7.3707
        p1_x,p1_y,p1_z=r0*np.cos(phi)*np.sin(np.pi/2.)/a+cent_point[0],r0*np.sin(phi)*np.sin(np.pi/2.)/b+cent_point[1],r0*np.cos(np.pi/2.)/c+cent_point[2]
        p2_x,p2_y,p2_z=r0*np.cos(phi+2*np.pi/3)*np.sin(np.pi/2.)/a+cent_point[0],r0*np.sin(phi+2*np.pi/3)*np.sin(np.pi/2.)/b+cent_point[1],r0*np.cos(np.pi/2.)/c+cent_point[2]
        p3_x,p3_y,p3_z=r0*np.cos(phi+4*np.pi/3)*np.sin(np.pi/2.)/a+cent_point[0],r0*np.sin(phi+4*np.pi/3)*np.sin(np.pi/2.)/b+cent_point[1],r0*np.cos(np.pi/2.)/c+cent_point[2]
        apex_x,apex_y,apex_z=cent_point[0],cent_point[1],cent_point[2]+r1/c
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=[apex_x,apex_y,apex_z])
        _add_sorbate(domain=domain,id_sorbate=O_ids[0],el='O',sorbate_v=[p1_x,p1_y,p1_z])
        _add_sorbate(domain=domain,id_sorbate=O_ids[1],el='O',sorbate_v=[p2_x,p2_y,p2_z])
        _add_sorbate(domain=domain,id_sorbate=O_ids[2],el='O',sorbate_v=[p3_x,p3_y,p3_z])
        
    def outer_sphere_complex2(self,domain,cent_point=[0.5,0.5,1.],r0=1.,r1=1.,phi1=0.,phi2=0.,theta=1.57,pb_id='pb1',O_ids=['Os1','Os2','Os3']):
        #different from version 1:consider the orientation of the pyramid, not just up and down
        #add a regular trigonal pyramid motiff above the surface representing the outer sphere complexation
        #the pyramid is oriented either Oxygen base top (when r1 is negative) or apex top (when r1 is positive)
        #cent_point is the fractional coordinates
        #r0 in ansgtrom is the distance between cent_point and oxygens in the based
        #r1 in angstrom is the distance bw cent_point and apex
        
        #anonymous function f1 calculating transforming matrix with the basis vector expressions,x1y1z1 is the original basis vector
        #x2y2z2 are basis of new coor defined in the original frame,new=T.orig
        f1=lambda x1,y1,z1,x2,y2,z2:np.array([[np.dot(x2,x1),np.dot(x2,y1),np.dot(x2,z1)],\
                                      [np.dot(y2,x1),np.dot(y2,y1),np.dot(y2,z1)],\
                                      [np.dot(z2,x1),np.dot(z2,y1),np.dot(z2,z1)]])
        #anonymous function f2 to calculate the distance bt two vectors
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        a0_v,b0_v,c0_v=np.array([1.,0.,0.]),np.array([0.,1.,0.]),np.array([0.,0.,1.]) 
        a,b,c=5.038,5.434,7.3707
        cell=np.array([a,b,c])
        cent_point=cell*cent_point
        p0=np.array(cent_point)
        #first step compute p1, use the original spherical frame origin at center point
        p1_x,p1_y,p1_z=r0*np.cos(phi1)*np.sin(theta)+cent_point[0],r0*np.sin(phi1)*np.sin(theta)+cent_point[1],r0*np.cos(theta)+cent_point[2]
        p1=np.array([p1_x,p1_y,p1_z])
        #step two setup spherical coordinate sys origin at p0
        z_v=(p1-p0)/f2(p0,p1)
        #working on the normal plane, it will crash if z_v[2]==0, check ppt file for detail algorithm
        temp_pt=None
        if z_v[2]!=0:
            temp_pt=np.array([0.,0.,(z_v[1]*p0[1]-z_v[0]*p0[0])/z_v[2]+p0[2]])
        elif z_v[1]!=0:
            temp_pt=np.array([0.,(z_v[2]*p0[2]-z_v[0]*p0[0])/z_v[1]+p0[1],0.])
        else:
            temp_pt=np.array([(-z_v[2]*p0[2]-z_v[1]*p0[1])/z_v[0]+p0[0],0.,0.])
        x_v=(temp_pt-p0)/f2(temp_pt,p0)
        y_v=np.cross(z_v,x_v)
        T=f1(a0_v,b0_v,c0_v,x_v,y_v,z_v)
        #then calculte p2, note using the fact p2p0 is 120 degree apart from p1p0, since the base is equilayer triangle
        p2_x,p2_y,p2_z=r0*np.cos(phi2)*np.sin(np.pi*2./3.),r0*np.sin(phi2)*np.sin(np.pi*2./3.),r0*np.cos(np.pi*2./3.)
        p2_new=np.array([p2_x,p2_y,p2_z])
        p2=np.dot(inv(T),p2_new)+p0
        #step three calculate p3, use the fact p3 on the vector extension of p1p2cent_p0
        p3=(p0-(p1+p2)/2.)*3+(p1+p2)/2.
        #step four calculate p4, cross product, note the magnitute here is in angstrom, so be careful
        p4_=np.cross(p2-p0,p1-p0)
        zero_v=np.array([0,0,0])
        p4=p4_/f2(p4_,zero_v)*r1+p0
        
        def _add_sorbate(domain=None,id_sorbate=None,el='Pb',sorbate_v=[]):
            sorbate_index=None
            try:
                sorbate_index=np.where(domain.id==id_sorbate)[0][0]
            except:
                domain.add_atom( id_sorbate, el,  sorbate_v[0] ,sorbate_v[1], sorbate_v[2] ,0.5,     1.00000e+00 ,     1.00000e+00 )
            if sorbate_index!=None:
                domain.x[sorbate_index]=sorbate_v[0]
                domain.y[sorbate_index]=sorbate_v[1]
                domain.z[sorbate_index]=sorbate_v[2]
        _add_sorbate(domain=domain,id_sorbate=pb_id,el='Pb',sorbate_v=p4/cell)
        _add_sorbate(domain=domain,id_sorbate=O_ids[0],el='O',sorbate_v=p1/cell)
        _add_sorbate(domain=domain,id_sorbate=O_ids[1],el='O',sorbate_v=p2/cell)
        _add_sorbate(domain=domain,id_sorbate=O_ids[2],el='O',sorbate_v=p3/cell)      
    
    def create_match_lib(self,domain,id_list):
        basis=np.array([5.038,5.434,7.3707])
        match_lib={}
        for i in id_list:
            match_lib[i]=[]
        f1=lambda domain,index:np.array([domain.x[index],domain.y[index],domain.z[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        #index=np.where(domain.id==center_atom_id)[0][0]
        for i in range(len(id_list)):
            index_1=np.where(domain.id==id_list[i])[0][0]
            for j in range(len(domain.id)):
                index_2=np.where(domain.id==domain.id[j])[0][0]
                if (f2(f1(domain,index_1),f1(domain,index_2))<2.5):
                    print f2(f1(domain,index_1),f1(domain,index_2))
                    match_lib[id_list[i]].append(domain.id[j])
        return match_lib
        
    def cal_bond_valence3(self,domain,match_lib):
        #match_lib={'O1':[['Fe1','Fe2'],['-x','+y']]}
        #calculate the bond valence of (in this case) O1_Fe1, O1_Fe2, where Fe1 and Fe2 have offset defined by '-x' and '+y' respectively.
        #return a lib with the same key as match_lib, the value for each key is the bond valence calculated
        bond_valence_container={}
        for i in match_lib.keys():
            match_lib[i].append(0)
            bond_valence_container[i]=0
            
        basis=np.array([5.038,5.434,7.3707])
        f1=lambda domain,index:np.array([domain.x[index]+domain.dx1[index],domain.y[index]+domain.dy1[index],domain.z[index]+domain.dz1[index]])*basis
        f2=lambda p1,p2:np.sqrt(np.sum((p1-p2)**2))
        
        def _offset_translate(flag):
            if flag=='+x':
                return np.array([1.,0.,0.])*basis
            elif flag=='-x':
                return np.array([-1.,0.,0.])*basis
            elif flag=='+y':
                return np.array([0.,1.,0.])*basis
            elif flag=='-y':
                return np.array([0.,-1.,0.])*basis
            elif flag=='+x+y':
                return np.array([1.,1.,0.])*basis
            elif flag=='+x-y':
                return np.array([1.,-1.,0.])*basis
            elif flag=='-x-y':
                return np.array([-1.,-1.,0.])*basis
            elif flag=='-x+y':
                return np.array([-1.,1.,0.])*basis
            elif flag==None:
                return np.array([0.,0.,0.])*basis
    
        for i in match_lib.keys():
            index=np.where(domain.id==i)[0][0]
            for k in range(len(match_lib[i][0])):
                j=match_lib[i][0][k]
                index2=np.where(domain.id==j)[0][0]
                dist=f2(f1(domain,index),f1(domain,index2)+_offset_translate(match_lib[i][1][k]))
                r0=0
                if (domain.el[index]=='Pb')|(domain.el[index2]=='Pb'):r0=2.112
                elif (domain.el[index]=='Fe')|(domain.el[index2]=='Fe'):r0=1.759
                elif (domain.el[index]=='Sb')|(domain.el[index2]=='Sb'):r0=1.973
                elif (domain.el[index]=='O')&(domain.el[index2]=='O'):#when two Oxygen atoms are too close, the structure explose with high r0, so we are expecting a high bond valence value here.
                    if dist<2.3:r0=10
                    else:r0=0.
                #if (i=='pb1'):
                    #print j,str(match_lib[i][1][k]),dist,'pb_coor',f1(domain,index)/basis,'O_coor',(f1(domain,index2)+_offset_translate(match_lib[i][1][k]))/basis,np.exp((r0-dist)/0.37)
                if dist<3.:#take it counted only when they are not two far away
                    bond_valence_container[i]=bond_valence_container[i]+np.exp((r0-dist)/0.37)
                    match_lib[i][2]=match_lib[i][2]+1
        for i in bond_valence_container.keys():
            #try to add hydrogen or hydrogen bond to the oxygen with 1.6=2*OH, 1.=OH+H, 0.8=OH and 0.2=H
            index=np.where(domain.id==i)[0][0]
            if (domain.el[index]=='O')|(domain.el[index]=='o'):
                case_tag=match_lib[i][2]
                bond_valence_corrected_value=[0.]
                if case_tag==1.:
                    bond_valence_corrected_value=[1.8,1.6,1.2,1.,0.8,0.6,0.4,0.2,0.]
                elif case_tag==2.:
                    bond_valence_corrected_value=[1.6,1.,0.8,0.4,0.2,0.]
                elif case_tag==3.:
                    bond_valence_corrected_value=[0.8,0.2,0.]
                else:pass
                #bond_valence_corrected_value=[1.6,1.,0.8,0.2,0.]
                ref=np.sign(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)*(bond_valence_container[i]+np.array(bond_valence_corrected_value)-2.)
                bond_valence_container[i]=bond_valence_container[i]+bond_valence_corrected_value[np.where(ref==np.min(ref))[0][0]]

        return bond_valence_container
                