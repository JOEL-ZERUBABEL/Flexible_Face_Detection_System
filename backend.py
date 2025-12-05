import joblib
from scipy.spatial import distance
from deepface import DeepFace
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_CONTOURS,FACEMESH_IRISES,FACEMESH_TESSELATION
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
import mediapipe as mp
from cvzone.FaceDetectionModule import FaceDetector
from sklearn.metrics.pairwise import cosine_similarity

try:
    from embedding_utils import EmbeddingUtils
except Exception:
    EmbeddingUtils=None
from numpy import ndarray
from typing import Optional,List,Tuple,Dict


class Face_Detection:
    def __init__(self):
        self.deepface=DeepFace.build_model(model_name='ArcFace')
        self.facedetector=mp.solutions.face_detection.FaceDetection(model_selection=0,min_detection_confidence=0.5)
        self.mp_face_mesh=mp.solutions.face_mesh
        self.mp_drawings=mp.solutions.drawing_utils
        self.drawing_spec=self.mp_drawings.DrawingSpec(thickness=1,circle_radius=1)
        self.facemesh=self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5,min_tracking_confidence=0.5,max_num_faces=1,refine_landmarks=True)
        self.embedding=EmbeddingUtils() if EmbeddingUtils is not None else None
        try:
            self.load=joblib.load('face_detection.pkl')
        except Exception:
            self.load=None
    @staticmethod
    def rgb(img:np.ndarray)->ndarray:
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def bgr(img:np.ndarray)->ndarray:
        return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
        
    def normalize(self,box,image_w,image_h)->Tuple[int,int,int,int]:
        x1=int(0,box[0]*image_w)
        x2=int(0,box[1]*image_h)
        y1=int(min(0,box[2]*image_w))
        y2=int(min(0,box[3]*image_h))
        return x1,y1,x2,y2     
    
    def lbp(self,img):
        radius=2
        n_points=8*radius
        if len(img.shape)==3:
            image_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        else:
            image_gray=img
        lbp=local_binary_pattern(img,n_points,radius)  
        return lbp
    
    

    
    def _get_deepface_embedding(self, img) -> Optional[np.ndarray]:
        try:
            if not isinstance(img, np.ndarray):
                return None
        
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rep = DeepFace.represent(
                img_path=rgb_img,
                model_name='ArcFace',
                enforce_detection=True,      
                detector_backend='mediapipe'
            )
            if isinstance(rep, list) and len(rep) > 0:
                emb = rep[0].get("embedding", None)
                if emb is None:
                    return None
                return np.asarray(emb, dtype=np.float32)

            return None

        except Exception as e:
            print("Embedding ERROR:", e)
            return None


    
    def web_verify(self,img1:np.ndarray,img2:np.ndarray)->Dict[str,float]:
        embone=self._get_deepface_embedding(img1)
        embtwo=self._get_deepface_embedding(img2)
        if embone is None or embtwo is None:
            return {'similarity':0.0,"euclidean":float('inf'),'status':'error'}
        cos_sim = float(cosine_similarity([embone], [embtwo])[0][0])

        euc=float(distance.euclidean(embone,embtwo))          
        dist=0.45
        
        is_match=cos_sim>(1-dist)
        status='match' if is_match else 'no match'
        return {"similarity": cos_sim, "euclidean": euc, "status": status}
        
    def photo_verify(self,img_path:str)->Optional[Dict]:
        try:
            img=cv2.imread(img_path)
            if img_path is None:
                print('no image found')
                return None
            bboxes,_=self.facedetector.findFaces(img,draw=False)
            if not bboxes:
                print('no face detected')
                return{'status':'none'}
            box0=bboxes[0]
            if isinstance(box0,dict):
                if 'bboxes' in box0:
                    x,y,w,h=box0['bboxes']
                elif 'bbox' in box0:
                    x,y,w,h=box0['boxes']
                else:
                    x=box0.get('x',0)
                    y=box0.get('y',0)
                    w=box0.get('w',img.shape[1]-x)
                    h=box0.get('h',img.shape[1]-y) 
            else:
                x,y,w,h=box0
            face=img[y:y+h,x:x+w]
            emb=self._get_deepface_embedding(face)
            if emb is None:
                return {'status':'no embedding'}
            stored=self.load
            if stored is None:
                return {'status':'no stored embedding'}
            if isinstance(stored,dict) and 'embedding' in stored:
                stored_arr=np.asarray(stored['embedding'],dtype=np.float32),np.reshape(1,-1)
            else:
                stored_arr=np.asanyarray(stored)
                if stored_arr.ndim==1:
                    stored_arr=stored_arr.reshape(1,-1)
            cos_sim=cosine_similarity([emb],stored_arr)[0]
            best_idx=int(np.argmax(cos_sim))
            best_score=float(cos_sim[best_idx])
            eucl=[float(distance.euclidean(emb,s)) for s in stored_arr]
            result={'best score':best_score,
                    'best index':best_idx,
                    'euclidean':eucl[best_idx],
                    'all score':cos_sim.tolist(),
                    'status':'status match' if best_score>0.5 else 'no match'}
            return result

        except Exception as e:
            print('photo verified',e)
            return None
                

    def blink_ear_detection(self,frame:np.ndarray)->Dict[str,int]:
        EAR_THRESHOLD=0.25
        BLINK_CONSENC_FRAMES=2
        blink_count=0
        frame_rate=0
        LEFT_EYE=[33,160,158,133,153,144]
        RIGHT_EYE=[263,387,385,262,380,373]

        rgb=self.rgb(frame)
        result=self.facemesh.process(rgb)
        if not result.multi_face_landmarks:
            return{'blink count':0,'ear':0.0,'detected':False}
        face_landmarks=result.multi_face_landmarks[0]
        h,w=frame.shape[:2]

        def lm_to_point(idx):
            lm=face_landmarks.landmark[idx]
            return (int(lm.x*w),int(lm.y*h))
        left_pts=[lm_to_point(i) for i in LEFT_EYE]
        right_pts=[lm_to_point(i) for i in RIGHT_EYE]
        

                    #for ear
        def ear(eye):
            p1,p2,p3,p4,p5,p6=eye
            A=distance.euclidean(p2,p6)
            B=distance.euclidean(p3,p5)
            C=distance.euclidean(p1,p4)
            if C==0:
                return 0.0
            return (A+B)/(2.0*C)
                    
                        
        left_ear=ear(left_pts)
        right_ear=ear(right_pts)
        avg_ear=(left_ear+right_ear)/2.0
        blink=1 if avg_ear<EAR_THRESHOLD else 0
        return{'blink count':blink,"ear":avg_ear,'detected':True}
    
                    

    def mouth_detection(self,frame):
        UPPER_LIP=13
        LOWER_LIP=14
        LEFT_LIP=78
        RIGHT_LIP=308
        rgb=self.rgb(frame)
        result=self.facemesh.process(rgb)
        if not result.multi_face_landmarks:
            return {'mar':0.0,'open':False,'detected':False}
        
        if result.multi_face_landmarks:
            lm=result.multi_face_landmarks[0].landmark
            h,w=frame.shape[:2]
        
        upper = (int(lm[UPPER_LIP].x * w), int(lm[UPPER_LIP].y * h))
        down = (int(lm[LOWER_LIP].x * w), int(lm[LOWER_LIP].y * h))
        left_mouth = (int(lm[LEFT_LIP].x * w), int(lm[LEFT_LIP].y * h))
        right_mouth = (int(lm[RIGHT_LIP].x * w), int(lm[RIGHT_LIP].y * h))
        
        vertical_distance=distance.euclidean(upper,down)
        horizontal_distance=distance.euclidean(left_mouth,right_mouth)
        mar=vertical_distance/horizontal_distance

        open_flag=mar>=0.65
        return{'mar':mar,'open':open_flag,'detected':True}

    def nose_detection(self,frame:np.ndarray)->Dict[str,float]:
        rgb=self.rgb(frame)
        result=self.facemesh.process(rgb)
        if not result.multi_face_landmarks:
            return {'detected':False}
        
        face_landmarks=result.multi_face_landmarks[0]
        h,w=frame.shape[:2]

        landmark_idx=[1,33,263,61,291,199]
        image_points=[]
        for idx in landmark_idx:
            lm=face_landmarks.landmark[idx]
            x,y=int(lm.x*w),int(lm.y*h)
            image_points.append((x,y))


        image_points=np.array(image_points,dtype='double')
        model_points=np.array([
            [0.0,0.0,0.0],
            [-30.0,-125.0,-30.0],
            [30.0,-125.0,-35.0],
            [-60.0,50.0,-30.0],
            [60.0,50.0,-30.0],
            [0.0,100.0,-30.0]
        ],dtype='double')

        focal_length=w
        center=(w/2,h/2)
        camera_matrix=np.array([
            [focal_length,0,center[0]],
            [0,focal_length,center[1]],
            [0,0,1]
        ],dtype='double')

        dist_coeff=np.zeros((4,1),dtype='double')
        success,rotation_vec,translation_vec=cv2.solvePnP(model_points,image_points,camera_matrix,dist_coeff,flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return{'detected':False,'frame':frame}
        
        rotation_matrix,_=cv2.Rodrigues(rotation_vec)
        sy=np.sqrt(rotation_matrix[0,0]**2+rotation_matrix[1,0]**2)
        

        pitch=np.arctan2(-rotation_matrix[2,0],sy*180.0/np.pi)
        yaw=np.arctan2(rotation_matrix[1,0],rotation_matrix[0,0])*180.0/np.pi
        roll=np.arctan2(rotation_matrix[2,1],rotation_matrix[2,2])*180.0/np.pi

        '''if pitch:
            cv2.putText(frame,'cant detect',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        if yaw:
            cv2.putText(frame,'cant detect',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)
        if roll:
            cv2.putText(frame,'cant detect',(10,60),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,0),2)'''

        return {'detected':True,
                'pitch':pitch,
                'yaw':yaw,'roll':roll,'frame':frame}
    
if __name__ == '__main__':
    fd = Face_Detection()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        fd.blink_ear_detection(frame)
        fd.mouth_detection(frame)
        fd.nose_detection(frame)

      
        cv2.imshow("Live Feed", frame)

        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
