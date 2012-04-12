import models.sxrd_test5_sym_new_test_new66_2 as model
from models.utils import UserVars
import numpy as np
from operator import mul
from numpy.linalg import inv
import sys
sys.path.append('Y:\\github\\polyhedra-geometry')
import hexahedra,tetrahedra,octahedra

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
    
    def build_super_cell(self,ref_domain,attach=''):
    #build a super cell based on the ref_domain, the super cell is actually two domains stacking together in x direction
        super_cell=ref_domain.copy()
        for id in ref_domain.id:
            index=np.where(ref_domain.id==id)[0][0]
            super_cell.insert_atom(index=index,id=str(id)+attach,element=ref_domain.el[index], x=ref_domain.x[index]+1.0, y=ref_domain.y[index], z=ref_domain.z[index], u = ref_domain.u[index], oc = ref_domain.oc[index], m = ref_domain.m[index])
        return super_cell
    
    def create_equivalent_domains(self):
    #each layer here has two atoms, so if you say terminate at oth layer (then id counted from 0, which is right)
    #if terminated at 1st layer (acutally 2nd layer since we count from 0), then Id will count from 2, which is also make sense
        new_domain_A=self.ref_domain.copy()
        new_domain_B=self.ref_domain.copy()
        for id in self.id_list[:self.terminated_layer*2]:
            if id!=[]:
                new_domain_A.del_atom(id)
        #number 5 here is crystal specific, here is the case for hematite
        for id in self.id_list[:self.terminated_layer*2+5*2]:
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
                polyhedra_index=[int(line_split[0]),int(line_split[1])]
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
            domain_list[i].x[id_index]=new_point_xyz[0]/5.038
            domain_list[i].y[id_index]=new_point_xyz[1]/5.434
            domain_list[i].z[id_index]=new_point_xyz[2]/7.3707
    
        
    def updata_sorbate_polyhedra2_batch(self,file):
        f=open(file)
        lines=f.readlines()
        for line in lines:
            if line[0]!='#':
                line_split=line.rsplit(',')
                domain_list=[vars(self)[line_split[0]],vars(self)[line_split[1]]]
                id_list=[line_split[2],line_split[3]]
                polyhedra_index_list=[int(line_split[4]),int(line_split[5])]
                offset=int(line_split[6])
                r=getattr(self.new_var_module,line_split[7])
                theta=getattr(self.new_var_module,line_split[8])
                phi=getattr(self.new_var_module,line_split[9])
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
    def add_oxygen_pair2(self,domain,ref_id,O_id,v_shift,r,alpha):
    #v_shift and r are in unit of angstrom
        ref_index=np.where(domain.id==ref_id)[0]
        basis=np.array([5.038,5.434,7.3707])
        ref_point=[domain.x[ref_index]*basis[0],domain.y[ref_index]*basis[1],domain.z[ref_index]*basis[2]+v_shift]
        x_shift=r*np.cos(alpha)
        y_shift=r*np.sin(alpha)
        point1=np.array([ref_point[0][0]-x_shift,ref_point[1][0]-y_shift,ref_point[2][0]])/basis
        point2=np.array([ref_point[0][0]+x_shift,ref_point[1][0]+y_shift,ref_point[2][0]])/basis
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
                elif line_split[0]=='sign':
                    tmp_list=[]
                    for i in range(len(line_split)-3):
                        tmp_list.append(self.norm_sign(getattr(self.new_var_module,line_split[i+2])))
                    setattr(self,line_split[1],tmp_list)
                    
                    