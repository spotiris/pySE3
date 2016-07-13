''' Provides convention - consistent operations and conversions between 
 quaternions, Euler angles and rotation matricies.
 IMPORTANT NOTE: This uses left handed convention, so negate all angles to get RH convention.
 I think this matches with the following document: 
 http://ocw.mit.edu/courses/mechanical-engineering/2-154-maneuvering-and-control-of-surface-and-underwater-vehicles-13-49-fall-2004/lecture-notes/lec1.pdf
 '''
from __future__ import print_function
# from sys import stderr
from math import sin, cos, atan2, asin, sqrt
import numpy as np


class EulerAngles(object):
  ''' Euler angles in [roll, pitch, yaw] order.
   Currently only rotation_order='ZYX' is supported.'''
  @staticmethod
  def from_rotation_matrix(rotation_matrix, rotation_order='ZYX'):
    '''Calculate Euler angles from a rotation matrix'''
    e = None
    if rotation_order == 'ZYX':
      e = np.array([[atan2(rotation_matrix[1, 2], rotation_matrix[2, 2])]
        , [-asin(rotation_matrix[0, 2])]
        , [atan2(rotation_matrix[0, 1], rotation_matrix[0, 0])]], dtype=np.float64)
    else:
      raise Exception("EulerAngles.from_rotation_matrix: cannot convert from rotation order '%s'" % (rotation_order))
    return Rotation(e, rotation_type='euler_angles', rotation_order=rotation_order)

  @staticmethod
  def from_quaternion(q, rotation_order='ZYX'):
    e = None
    if rotation_order == 'ZYX':
      e = np.array([[atan2(2 * q[0] * q[1] + 2 * q[2] * q[3], q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3])]
          ,[asin(2 * q[0] * q[2] - 2 * q[1] * q[3])]
          ,[atan2(2 * q[0] * q[3] + 2 * q[1] * q[2], q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3])]], dtype=np.float64)
    else:
      raise Exception("EulerAngles.from_quaternion: cannot convert from rotation order '%s'" % (rotation_order))
    return Rotation(e, rotation_type='euler_angles', rotation_order=rotation_order)

  @staticmethod
  def time_integral(angles, angular_velocity, dt, rotation_order='ZYX'):
    if rotation_order is not 'ZYX':
      raise Exception("EulerAngles.from_quaternion: cannot convert from rotation order '%s'" % (rotation_order))
    sin_theta, cos_theta = sin(angles[1]), cos(angles[1])
    sin_phi, cos_phi = sin(angles[2]), cos(angles[2])
    return angles + np.fliplr(np.array([
            [0, sin_phi, cos_phi],
            [0, cos_phi * cos_theta, -sin_phi * cos_theta],
            [cos_theta, sin_phi * sin_theta, cos_phi * sin_theta]
        ])).dot(angular_velocity * dt) / cos_theta
  @staticmethod
  def rates_to_angular_velocity(angles, euler_rates, rotation_order='ZYX'):
    sin_theta, cos_theta = sin(angles[1]), cos(angles[1])
    sin_phi, cos_phi = sin(angles[2]), cos(angles[2])
    return np.array([
        [cos_phi * cos_theta, -sin_phi , 0],
        [sin_phi * cos_theta, cos_phi, 0],
        [-sin_theta, 0, 1]
        ]).dot(euler_rates)

class Quaternion(object):
  ''' Quaternion in [w, x , y, z] order '''
  @staticmethod
  def from_euler_angles(euler, rotation_order='ZYX'):
    cr = cos(euler[0] / 2.0)
    sr = sin(euler[0] / 2.0)
    sp = sin(euler[1] / 2.0)
    cp = cos(euler[1] / 2.0)
    cy = cos(euler[2] / 2.0)
    sy = sin(euler[2] / 2.0)
    q = None
    if rotation_order == 'ZYX':
      q = np.array([[cp * cr * cy + sp * sr * sy] \
        , [cp * cy * sr - cr * sp * sy]\
        , [cr * cy * sp + cp * sr * sy]\
        , [cp * cr * sy - cy * sp * sr]], dtype=np.float64)
    else:
      raise Exception("Quaternion.from_euler_angles: cannot convert from rotation order '%s'" % (rotation_order))
    return Rotation(q, rotation_type='quaternion')
  @staticmethod
  def from_rotation_matrix(R):
    q = np.zeros((4, 1), dtype=np.float64)
    tr = R.trace()
    if tr > 0:
      sqtrp1 = sqrt(tr + 1.0)
      q[0] = 0.5 * sqtrp1
      q[1] = (R[1, 2] - R[2, 1]) / (2.0 * sqtrp1)
      q[2] = (R[2, 0] - R[0, 2]) / (2.0 * sqtrp1)
      q[3] = (R[0, 1] - R[1, 0]) / (2.0 * sqtrp1)
    else:
      d = R.diagonal()
      if (d[1] > d[0]) and (d[1] > d[2]):
        sqdip1 = sqrt(d[1] - d[0] - d[2] + 1.0)
        q[2] = 0.5 * sqdip1
        if sqdip1 != 0:
          sqdip1 = 0.5 / sqdip1
        q[0] = (R[2, 0] - R[0, 2]) * sqdip1
        q[1] = (R[0, 1] + R[1, 0]) * sqdip1
        q[3] = (R[1, 2] + R[2, 1]) * sqdip1
      elif d[2] > d[0]:
        sqdip1 = sqrt(d[2] - d[0] - d[1] + 1.0)
        q[3] = 0.5 * sqdip1
        if sqdip1 != 0:
          sqdip1 = 0.5 / sqdip1
        q[0] = (R[0, 1] - R[1, 0]) * sqdip1
        q[1] = (R[2, 0] + R[0, 2]) * sqdip1
        q[2] = (R[1, 2] + R[2, 1]) * sqdip1
      else:
        sqdip1 = sqrt(d[0] - d[1] - d[2] + 1.0)
        q[1] = 0.5 * sqdip1
        if sqdip1 != 0:
          sqdip1 = 0.5 / sqdip1
        q[0] = (R[1, 2] - R[2, 1]) * sqdip1
        q[2] = (R[0, 1] + R[1, 0]) * sqdip1
        q[3] = (R[2, 0] + R[0, 2]) * sqdip1
    return Rotation(q, rotation_type='quaternion')
  @staticmethod
  def normalize(q):
    if q[0] < 0:
      q *= -1.0
    return q / np.linalg.norm(q)
  @staticmethod
  def multiply(q, r):
    return \
    Rotation(np.array([[q[0] * r[0] - q[1] * r[1] - q[2] * r[2] - q[3] * r[3]]
      , [q[0] * r[1] + q[1] * r[0] + q[2] * r[3] - q[3] * r[2]]
      , [q[0] * r[2] + q[2] * r[0] - q[1] * r[3] + q[3] * r[1]]
      , [q[0] * r[3] + q[1] * r[2] - q[2] * r[1] + q[3] * r[0]]],
       dtype=np.float64).reshape((4, 1))
        , rotation_type='quaternion') #TODO: Why is reshape needed?
  @staticmethod
  def conjugate(q):
    Q = q.copy()
    Q[1:4] *= -1
    return Q
  @staticmethod
  def rotate_vector(q, v, reverse=False):
    V = np.vstack((0, v))
    q_ = q.copy()
    Q = Quaternion.conjugate(q)
    if reverse:
      tmp = q_
      q_ = Q
      Q = tmp
    return Quaternion.multiply(Q, Quaternion.multiply(V, q_))[1:4]
  @staticmethod
  def time_derivative(q, angular_rate):
    W = np.array([
      [0.0], [angular_rate[0]], [angular_rate[1]], [angular_rate[2]]
      ])
    return 0.5 * Quaternion.multiply(W, q)
  @staticmethod
  def time_integral(q, angular_rate, dt):
    norm = np.linalg.norm(angular_rate)
    angle = norm / 2.0 * dt
    r = np.vstack((cos(angle), angular_rate / norm * np.sin(angle)))
    return Quaternion.multiply(r, q)
  @staticmethod
  def from_vector(v2, v1):
    ''' return the rotation from v1 to v2 via the smallest arc '''
    v1_len = np.linalg.norm(v1)
    v2_len = np.linalg.norm(v2)
    v_orthogonal = np.cross(v1.ravel(), v2.ravel())
    q = Rotation.quaternion(
      np.dot(v1.ravel(), v2.ravel()) + np.sqrt((v1_len * v1_len) * (v2_len * v2_len)),
      v_orthogonal[0], v_orthogonal[1], v_orthogonal[2])
    return Quaternion.normalize(Quaternion.normalize(q))
  @staticmethod
  def from_axis_angle(axis, angle):
    ''' return a quaternion from an axis-angle representation '''
    half_angle = angle * 0.5
    s = np.sin(half_angle)
    return Quaternion.normalize( \
      Rotation.quaternion( \
        np.cos(half_angle), axis[0] * s, axis[1] * s, axis[2] * s))
class RotationMatrix(object):
  ''' Rotation matrix (Direction Cosine Matrix, DCM) '''
  @staticmethod
  def from_euler_angles(euler, rotation_order='ZYX'):
    cr = cos(euler[0])
    sr = sin(euler[0])
    sp = sin(euler[1])
    cp = cos(euler[1])
    cy = cos(euler[2])
    sy = sin(euler[2])
    R = None
    if rotation_order == 'ZYX':
      R = np.array(
        [[cp * cy, cp * sy, -sp], 
        [cy * sp * sr - cr * sy, cr * cy + sp * sr * sy, cp * sr], 
        [sr * sy + cr * cy * sp, cr * sp * sy - cy * sr, cp * cr]],
        dtype=np.float64)
    else:
      raise Exception("Quaternion.from_euler_angles: cannot convert from rotation order '%s'" % (rotation_order))
    return R
  @staticmethod
  def from_quaternion(q):
    q0s = q[0] * q[0]
    q1s = q[1] * q[1]
    q2s = q[2] * q[2]
    q3s = q[3] * q[3]
    q0q3 = q[0] * q[3]
    q1q2 = q[1] * q[2]
    q0q2 = q[0] * q[2]
    q1q3 = q[1] * q[3]
    q0q1 = q[0] * q[1]
    q2q3 = q[2] * q[3]
    n = q0s + q1s + q2s + q3s
    return np.array(
      [[q0s + q1s - q2s - q3s, 2 * (q0q3 + q1q2), -2 * (q0q2 - q1q3)], 
      [-2 * (q0q3 - q1q2), q0s - q1s + q2s - q3s, 2 * (q0q1 + q2q3)], 
      [2 * (q0q2 + q1q3), -2 * (q0q1 - q2q3), q0s - q1s - q2s + q3s]], 
      dtype=np.float64).reshape(3, 3) / n

class Rotation(np.ndarray):
  ''' Factory for all rotation and orientation conversions and construtors '''
  @staticmethod
  def euler_angles(*args, **kwargs):
    ''' converts Rotation objects to euler angles, or constructs
     them from (roll, pitch, yaw) angles '''
    n_args = len(args)
    rotation_order='ZYX'
    vec = None
    if 'rotation_order' in kwargs:
      rotation_order = kwargs['rotation_order']
    if not rotation_order in ['ZYX']:
      raise Exception("Rotation.euler_angles: Unsupported rotation order '%s'"
        % str(args[3]))

    if n_args == 0:
      vec = np.array([[0.0],[0.0],[0.0]], dtype=np.float64)
    elif n_args == 1 and type(args[0]) is np.ndarray: # numpy array
      vec = args[0].ravel()
    elif n_args == 1 and type(args[0]) is Rotation: # rotation_object
      rotation = args[0]
      if rotation.rotation_type == 'rotation_matrix':
        vec = EulerAngles.from_rotation_matrix(rotation, rotation_order)
      elif rotation.rotation_type == 'quaternion':
        vec = EulerAngles.from_quaternion(rotation, rotation_order)
      else:
        raise Exception("Rotation.euler_angles: unsupported rotation type constructor")
    elif n_args == 3: # x, y , z[,rotation_order]
      vec = np.array([[args[0]],[args[1]],[args[2]]], dtype=np.float64)
    else:
      raise Exception("Rotation.euler_angles: Unknown constructor arguments")
    return Rotation(vec, rotation_type='euler_angles', rotation_order=rotation_order)
      
  @staticmethod
  def matrix(rotation):
    ''' Converts a Rotation object into a rotation matrix
    (Direction Cosine Matrix) or construct one from a [3, 3] ndarray '''
    R = None
    if type(rotation) is np.ndarray and rotation.shape == (3, 3):
      R = rotation
    elif type(rotation) is Rotation:
      if rotation.rotation_type == 'euler_angles':
        R = RotationMatrix.from_euler_angles(rotation
          , rotation_order=rotation.rotation_order)
      elif rotation.rotation_type == 'quaternion':
        R = RotationMatrix.from_quaternion(rotation)
      else:
        raise Exception("Rotation.matrix: cannot convert unknown rotation object of type '%s'" % (rotation.rotation_type))
    else:
      raise Exception("Rotation.matrix: cannot convert unknown object in constructor")
    return Rotation(R, rotation_type='rotation_matrix')

  @staticmethod
  def quaternion(*args):
    ''' Convert a Rotation object to a quaternion, or construct one
     using a 4 - size ndarray, or arguments (w, x , y, z).
     Quaternion vector order is [w, x , y, z]' with shape [4, 1]'''
    nargs = len(args)
    q = None
    if nargs == 4:
      q = np.array([args[0], args[1], args[2], args[3]]).reshape(4, 1)
    else:
      rotation = args[0]
      if type(rotation) is np.ndarray and rotation.size == 4:
        q = rotation.reshape([4, 1])
      elif type(rotation) is Rotation:
        if rotation.rotation_type == 'quaternion':
          q = args[0]
        elif rotation.rotation_type == 'euler_angles':
          q = Quaternion.from_euler_angles(rotation, 
            rotation_order=rotation.rotation_order)
        elif rotation.rotation_type == 'rotation_matrix':
          q = Quaternion.from_rotation_matrix(rotation)
        else:
          raise Exception("Rotation.quaternion: cannot convert unknown rotation object of type '%s'" % (rotation.rotation_type))
      else:
        raise Exception("Rotation.quaternion: expected a Rotation object as argument")
    return Rotation(q, rotation_type='quaternion')

  def __new__(cls, input_array=None, rotation_type=None, rotation_order=None):
    obj = np.asarray(input_array).view(cls)
    # add the new attribute to the created instance
    obj.rotation_type = rotation_type
    obj.rotation_order = rotation_order
    return obj
  def __array_finalize__(self, obj):
    # see rotation_typeArray.__array_finalize__ for comments
    if obj is None:
      return
    self.rotation_type = getattr(obj, 'rotation_type', None)
    self.rotation_order = getattr(obj, 'rotation_order', None)

class Pose(object):
  ''' class to store and operate on poses'''
  def __init__(self, *args, **kwargs):
    '''construct a pose from a different types of arguments'''
    nargs = len(args)
    if nargs == 1 and isinstance(args[0], np.ndarray):
      pose = args[0]
      self.translation = np.array([
        [pose['x'][0]], [pose['y'][0]], [pose['z'][0]]])
      self.rotation = Rotation.quaternion(Rotation.euler_angles(
        pose['roll'][0], pose['pitch'][0], pose['yaw'][0]))
    elif nargs == 2 and isinstance(args[1], Rotation):
      # construct a pose from a translation and Rotation object
      self.translation = args[0].ravel().reshape((3,1))
      self.rotation = Rotation.quaternion(args[1])
    else:
      raise Exception("Pose: Unknown initialiser signature")
    self.scale = 1.0
    if 'scale' in kwargs:
      self.scale = kwargs['scale']
  @staticmethod
  def compose_point(pose, point, pad=False):
    ''' return the composition of a point and a pose '''
    T = np.zeros((4, 4))
    T[0:3, 0:3] = Rotation.matrix(pose.rotation) * pose.scale
    T[0:3, 3] = pose.translation 
    T[3, 3] = 1.0
    if pad:
      return T.dot(np.vstack((point, np.ones(point.shape[1]))))
    return T.dot(point)
  @staticmethod
  def compose_pose(pose, pose2):
    ''' return the composition of a point and a pose '''
    p3 = pose.translation + \
      Quaternion.rotate_vector(pose.rotation, pose2.translation)
    q3 = Quaternion.multiply(Rotation.quaternion(pose.rotation), \
      Rotation.quaternion(pose2.rotation))
    return Pose(p3, q3)
  def __add__(self, other):
    if isinstance(other, np.ndarray) and other.shape[0] == 3: # pose + point
      return Pose.compose_point(self, other)
    elif isinstance(other, Pose):
      return Pose.compose_pose(self, other)
    raise Exception("Pose.__add__: object not supported")
  def inverse(self):
    ''' Return an instance of the inverse pose'''
    return Pose(-self.translation, Quaternion.conjugate(self.rotation), scale=(1.0 / self.scale))

#################################################
def test():
  '''Test the rotation_formalisms module'''
  e = Rotation.euler_angles(-0.125, 0.25, 1.653, rotation_order='ZYX')
  q = Rotation.quaternion(e)
  R = Rotation.matrix(e)
  print("Euler angles:")
  print(e)
  print(Rotation.euler_angles(q))
  print(Rotation.euler_angles(R))
  print("Quaternion:")
  print(q)
  print(Rotation.quaternion(e))
  print(Rotation.quaternion(R))
  print("Rotation Matrix")
  print(R)
  print(Rotation.matrix(e))
  print(Rotation.matrix(q))

  print("Quaternion - vector rotation")
  vec = np.array([1.3231, -0.5174, 0.0243]).reshape(3, 1)
  q_ = Rotation.quaternion(0.9859, -0.0298, -0.1531, 0.0609)
  print(Quaternion.rotate_vector(q_, vec))
  print("reverse")
  print(Quaternion.rotate_vector(q_, vec, reverse=True))

  print("Quaternion integral")
  q = Rotation.quaternion(Rotation.euler_angles(0.0, 0.0, 0.0))
  w = np.array([[0], [0.75], [0.0]])
  dt = 0.1
  T = 1.0
  for t in np.linspace(0, T, T / dt):
    q = Quaternion.time_integral(q, w, dt)
    e = EulerAngles.from_quaternion(q)
    print(t, e.T)

if __name__ == "__main__":
  test()
  
