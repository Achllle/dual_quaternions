import os
import unittest
from pyquaternion import Quaternion
from dual_quaternions import DualQuaternion
import numpy as np

class TestDualQuaternion(unittest.TestCase):

    def setUp(self):
        self.identity_dq = DualQuaternion.identity()
        self.random_dq = DualQuaternion.from_quat_pose_array(np.array([1,2,3,4,5,6,7]))
        self.other_random_dq = DualQuaternion.from_quat_pose_array(np.array([0.2,0.1,0.3,0.07,1.2,0.9,0.2]))
        self.normalized_dq = self.random_dq.normalized()

    def test_creation(self):
        # from dual quaternion array: careful, need to supply a normalized DQ
        dql = np.array([0.7071067811, 0.7071067811, 0, 0, -3.535533905, 3.535533905, 1.767766952, -1.767766952])
        dq1 = DualQuaternion.from_dq_array(dql)
        dq2 = DualQuaternion.from_dq_array(dql)
        self.assertEqual(dq1, dq2)
        # from quaternion + translation array
        dq3 = DualQuaternion.from_quat_pose_array(np.array([1, 2, 3, 4, 5, 6, 7]))
        dq4 = DualQuaternion.from_quat_pose_array([1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(dq3, dq4)
        # from homogeneous transformation matrix
        T = np.array([[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, 1], [0, 0, 0, 1]])
        dq7 = DualQuaternion.from_homogeneous_matrix(T)
        self.assertEqual(dq7.q_r, Quaternion())
        self.assertEqual(dq7.translation(), [2, 3, 1])
        try:
            np.testing.assert_array_almost_equal(dq7.homogeneous_matrix(), T)
        except AssertionError as e:
            self.fail(e)
        # from a point
        dq8 = DualQuaternion.from_translation_vector([4, 6, 8])
        self.assertEqual(dq8.translation(), [4, 6, 8])

    def test_unit(self):
        q_r_unit = Quaternion(1, 0, 0, 0)
        q_d_zero = Quaternion(0, 0, 0, 0)
        unit_dq = DualQuaternion(q_r_unit, q_d_zero)
        self.assertEqual(self.identity_dq, unit_dq)
        # unit dual quaternion multiplied with another unit quaternion should yield unit
        self.assertEqual(self.identity_dq * self.identity_dq, self.identity_dq)

    def test_add(self):
        dq1 = DualQuaternion.from_translation_vector([4, 6, 8])
        dq2 = DualQuaternion.from_translation_vector([1, 2, 3])
        sum = dq1 + dq2
        self.assertEqual(sum.q_d, Quaternion(0., 2.5, 4., 5.5))

    def test_mult(self):
        # quaternion multiplication. Compare with homogeneous transformation matrices
        theta1 = np.pi / 180 * 20  # 20 deg
        T_pure_rot = np.array([[1., 0., 0., 0.],
                               [0., np.cos(theta1), -np.sin(theta1), 0.],
                               [0., np.sin(theta1), np.cos(theta1), 0.],
                               [0., 0., 0., 1.]])
        dq_pure_rot = DualQuaternion.from_homogeneous_matrix(T_pure_rot)
        T_pure_trans = np.array([[1., 0., 0., 1.],
                                 [0., 1., 0., 2.],
                                 [0., 0., 1., 3.],
                                 [0., 0., 0., 1.]])
        dq_pure_trans = DualQuaternion.from_homogeneous_matrix(T_pure_trans)

        T_double_rot = np.dot(T_pure_rot, T_pure_rot)
        dq_double_rot = dq_pure_rot * dq_pure_rot
        try:
            np.testing.assert_array_almost_equal(T_double_rot, dq_double_rot.homogeneous_matrix())
        except AssertionError as e:
            self.fail(e)

        T_double_trans = np.dot(T_pure_trans, T_pure_trans)
        dq_double_trans = dq_pure_trans * dq_pure_trans
        try:
            np.testing.assert_array_almost_equal(T_double_trans, dq_double_trans.homogeneous_matrix())
        except AssertionError as e:
            self.fail(e)

        # composed: trans and rot
        T_composed = np.dot(T_pure_rot, T_pure_trans)
        dq_composed = dq_pure_rot * dq_pure_trans
        dq_composed = dq_pure_rot * dq_pure_trans
        try:
            np.testing.assert_array_almost_equal(T_composed, dq_composed.homogeneous_matrix())
        except AssertionError as e:
            self.fail(e)

    def test_div(self):
        try:
            np.testing.assert_array_almost_equal((self.random_dq/self.random_dq).dq_array(),
                                                 self.identity_dq.dq_array())
            np.testing.assert_array_almost_equal((self.random_dq/self.identity_dq).dq_array(),
                                                 self.random_dq.dq_array())
        except AssertionError as e:
            self.fail(e)

    def test_inverse(self):
        # use known matrix inversion
        T_1_2 = np.array([[0, 1, 0, 2], [-1, 0, 0, 4], [0, 0, 1, 6], [0, 0, 0, 1]])
        T_2_1 = np.array([[0, -1, 0, 4], [1, 0, 0, -2], [0, 0, 1, -6], [0, 0, 0, 1]])
        dq_1_2 = DualQuaternion.from_homogeneous_matrix(T_1_2)
        dq_2_1 = DualQuaternion.from_homogeneous_matrix(T_2_1)

        try:
            np.testing.assert_array_almost_equal(dq_2_1.homogeneous_matrix(), dq_1_2.inverse().homogeneous_matrix())
        except AssertionError as e:
            self.fail(e)

    def test_inv_conj_unit(self):
        """For a unit dq, the inverse is equal to the quaternion conjugate"""
        self.assertEqual(self.normalized_dq.inverse(), self.normalized_dq.quaternion_conjugate())

    def test_equal(self):
        self.assertEqual(self.identity_dq, DualQuaternion.identity())
        self.assertEqual(self.identity_dq, DualQuaternion(-Quaternion(1, 0, 0, 0), -Quaternion(0, 0, 0, 0)))
        self.assertFalse(self.identity_dq == DualQuaternion(Quaternion(1, 0, 0, 1), -Quaternion(0, 0, 0, 0)))
        theta1 = np.pi / 180 * 20  # 20 deg
        T_pure_rot = np.array([[1., 0., 0., 0.],
                               [0., np.cos(theta1), -np.sin(theta1), 0.],
                               [0., np.sin(theta1), np.cos(theta1), 0.],
                               [0., 0., 0., 1.]])
        dq_pure_rot = DualQuaternion.from_homogeneous_matrix(T_pure_rot)
        # manually flip sign on terms
        dq_pure_rot.q_r = - dq_pure_rot.q_r
        dq_pure_rot.q_d = - dq_pure_rot.q_d
        try:
            np.testing.assert_array_almost_equal(dq_pure_rot.homogeneous_matrix(), T_pure_rot)
        except AssertionError as e:
            self.fail(e)
        dq_pure_rot.q_d = - dq_pure_rot.q_d
        try:
            np.testing.assert_array_almost_equal(dq_pure_rot.homogeneous_matrix(), T_pure_rot)
        except AssertionError as e:
            self.fail(e)
        dq_pure_rot.q_r = - dq_pure_rot.q_r
        try:
            np.testing.assert_array_almost_equal(dq_pure_rot.homogeneous_matrix(), T_pure_rot)
        except AssertionError as e:
            self.fail(e)

    def test_str_repr_is_string(self):
        # test that __str__ and __repr__ are working
        self.assertTrue(isinstance(repr(self.identity_dq), str))
        self.assertTrue(isinstance(self.identity_dq.__str__(), str))

    def test_quaternion_conjugate(self):
        dq = self.normalized_dq * self.normalized_dq.quaternion_conjugate()
        # a normalized quaternion multiplied with its quaternion conjugate should yield 1
        self.assertEqual(dq, DualQuaternion.identity())

        # test that the conjugate corresponds to the inverse of it's matrix representation
        matr = self.normalized_dq.homogeneous_matrix()
        inv = np.linalg.inv(matr)
        self.assertEqual(DualQuaternion.from_homogeneous_matrix(inv), self.normalized_dq.quaternion_conjugate())

        # (dq1 @ dq2)* ?= dq2* @ dq1*
        res1 = (self.random_dq * self.other_random_dq).quaternion_conjugate()
        res2 = self.other_random_dq.quaternion_conjugate() * self.random_dq.quaternion_conjugate()
        self.assertEqual(res1, res2)

    def test_homogeneous_conversion(self):
        # 1. starting from a homogeneous matrix
        theta1 = np.pi/2  # 90 deg
        trans = [10., 5., 0.]
        H1 = np.array([[1., 0., 0., trans[0]],
                     [0., np.cos(theta1), -np.sin(theta1), trans[1]],
                     [0., np.sin(theta1), np.cos(theta1), trans[2]],
                     [0., 0., 0., 1.]])
        # check that if we convert to DQ and back to homogeneous matrix, we get the same result
        double_conv1 = DualQuaternion.from_homogeneous_matrix(H1).homogeneous_matrix()
        try:
            np.testing.assert_array_almost_equal(H1, double_conv1)
        except AssertionError as e:
            self.fail(e)
        # check that dual quaternions are also equal
        dq1 = DualQuaternion.from_homogeneous_matrix(H1)
        dq_double1 = DualQuaternion.from_homogeneous_matrix(double_conv1)
        self.assertEqual(dq1, dq_double1)

        # 2. starting from a DQ
        dq_trans = DualQuaternion.from_translation_vector([10, 5, 0])
        dq_rot = DualQuaternion.from_dq_array([np.cos(theta1 / 2), np.sin(theta1 / 2), 0, 0, 0, 0, 0, 0])
        dq2 = dq_trans * dq_rot
        # check that this is the same as the previous DQ
        self.assertEqual(dq2, dq1)
        # check that if we convert to homogeneous matrix and back, we get the same result
        double_conv2 = DualQuaternion.from_homogeneous_matrix(dq2.homogeneous_matrix())
        self.assertEqual(dq2, double_conv2)

    def test_dual_number_conjugate(self):
        # dual number conjugate doesn't behave as you would expect given its special definition
        # (dq1 @ dq2)* ?= dq1* @ dq2*  This is a different order than the other conjugates!
        res1 = (self.random_dq * self.other_random_dq).dual_number_conjugate()
        res2 = self.random_dq.dual_number_conjugate() * self.other_random_dq.dual_number_conjugate()
        self.assertEqual(res1, res2)

    def test_combined_conjugate(self):
        dq = self.normalized_dq * self.normalized_dq.combined_conjugate()
        # a normalized quaternion multiplied with its combined conjugate should yield unit rotation
        self.assertAlmostEqual(dq.q_r, Quaternion())
        # (dq1 @ dq2)* ?= dq2* @ dq1*
        res1 = (self.random_dq * self.other_random_dq).combined_conjugate()
        res2 = self.other_random_dq.combined_conjugate() * self.random_dq.combined_conjugate()
        self.assertEqual(res1, res2)

    def test_normalize(self):
        self.assertTrue(self.identity_dq.is_normalized)
        self.assertEqual(self.identity_dq.normalized(), self.identity_dq)
        unnormalized_dq = DualQuaternion.from_quat_pose_array([1, 2, 3, 4, 5, 6, 7])
        manual_normalized = unnormalized_dq / abs(unnormalized_dq)
        unnormalized_dq.normalize()  # now normalized!
        self.assertTrue(unnormalized_dq.is_normalized)
        self.assertEqual(unnormalized_dq, manual_normalized)
        unnormalized_dq = DualQuaternion.from_dq_array([2, 1, 2, 3, 4, 5, 6, 7])
        manual_normalized = unnormalized_dq / abs(unnormalized_dq)
        unnormalized_dq.normalize()  # now normalized!
        self.assertTrue(unnormalized_dq.is_normalized)
        self.assertEqual(unnormalized_dq, manual_normalized)

    def test_transform(self):
        # transform a point from one frame (f2) to another (f1)
        point_f2 = [1, 1, 0]
        self.assertEqual(self.identity_dq.transform_point(point_f2), point_f2)

        # test that quaternion transform and matrix transform yield the same result
        T_f1_f2 = np.array([[1, 0, 0, 2],
                            [0, 0.54028748, -0.8414805, 3],
                            [0, 0.8414805, 0.54028748, 1],
                            [0, 0, 0, 1]])
        dq_f1_f2 = DualQuaternion.from_homogeneous_matrix(T_f1_f2)

        # point is in f2, transformation will express it in f1
        point_f1_matrix = np.dot(T_f1_f2, np.expand_dims(np.array(point_f2 + [1]), 1))
        point_f1_dq = np.array(dq_f1_f2.transform_point(point_f2))
        try:
            np.testing.assert_array_almost_equal(point_f1_matrix[:3].T.flatten(), point_f1_dq.flatten(), decimal=3)
        except AssertionError as e:
            self.fail(e)

    def test_screw(self):
        # test unit
        l, m, theta, d = self.identity_dq.screw()
        self.assertEqual(d, 0)
        self.assertEqual(theta, 0)

        # test pure translation
        trans = [10, 5, 0]
        dq_trans = DualQuaternion.from_translation_vector(trans)
        l, m, theta, d = dq_trans.screw()
        self.assertAlmostEqual(d, np.linalg.norm(trans), 2)
        self.assertAlmostEqual(theta, 0)

        # test pure rotation
        theta1 = np.pi/2
        dq_rot = DualQuaternion.from_dq_array([np.cos(theta1 / 2), np.sin(theta1 / 2), 0, 0, 0, 0, 0, 0])
        l, m, theta, d = dq_rot.screw()
        self.assertAlmostEqual(theta, theta1)

        # test simple rotation and translation: rotate in the plane of a coordinate system with the screw axis offset
        # along +y. Rotate around z axis so that the coordinate system stays in the plane. Translate along z-axis
        theta2 = np.pi/2
        dq_rot2 = DualQuaternion.from_dq_array([np.cos(theta2 / 2), 0, 0, np.sin(theta2 / 2), 0, 0, 0, 0])
        dist_axis = 5.
        displacement_z = 3.
        dq_trans = DualQuaternion.from_translation_vector([dist_axis*np.sin(theta2), dist_axis*(1.-np.cos(theta2)),
                                                           displacement_z])
        dq_comb = dq_trans * dq_rot2
        l, m, theta, d = dq_comb.screw()
        try:
            # the direction of the axis should align with the z axis of the origin
            np.testing.assert_array_almost_equal(l, np.array([0, 0, 1]), decimal=3)
            # m = p x l with p any point on the line
            np.testing.assert_array_almost_equal(np.cross(np.array([[0, dist_axis, 0]]), l).flatten(), m)
        except AssertionError as e:
            self.fail(e)
        self.assertAlmostEqual(d, displacement_z)  # only displacement along z should exist here
        self.assertAlmostEqual(theta, theta2)  # the angles should be the same

    def test_from_screw(self):
        # construct an axis along the positive z-axis
        l = np.array([0, 0, 1])
        # pick a point on the axis that defines it's location
        p = np.array([-1, 0, 0])
        # moment vector
        m = np.cross(p, l)
        theta = np.pi/2
        d = 3.
        # this corresponds to a rotation around the axis parallel with the origin's z-axis through the point p
        # the resulting transformation should move the origin to a DQ with elements:
        desired_dq_rot = DualQuaternion.from_quat_pose_array([np.cos(theta/2), 0, 0, np.sin(theta/2), 0, 0, 0])
        desired_dq_trans = DualQuaternion.from_translation_vector([-1, 1, d])
        desired_dq = desired_dq_trans * desired_dq_rot
        dq = DualQuaternion.from_screw(l, m, theta, d)
        self.assertEqual(dq, desired_dq)

    def test_from_screw_and_back(self):
        # start with a random valid dual quaternion
        dq = DualQuaternion.from_quat_pose_array([0.5, 0.3, 0.1, 0.4, 2, 5, -2])
        lr, mr, thetar, dr = dq.screw()
        dq_reconstructed = DualQuaternion.from_screw(lr, mr, thetar, dr)
        self.assertEqual(dq, dq_reconstructed)

        # start with some screw parameters
        l1 = np.array([0.4, 0.2, 0.5])
        l1 /= np.linalg.norm(l1)  # make sure l1 is normalized
        # pick some point away from the origin
        p1 = np.array([2.3, 0.9, 1.1])
        m1 = np.cross(p1, l1)
        d1 = 4.32
        theta1 = 1.94
        dq1 = DualQuaternion.from_screw(l1, m1, theta1, d1)
        l2, m2, theta2, d2 = dq1.screw()
        try:
            np.testing.assert_array_almost_equal(l1, l2, decimal=3)
            np.testing.assert_array_almost_equal(l1, l2, decimal=3)
        except AssertionError as e:
            self.fail(e)
        self.assertAlmostEqual(theta1, theta2)
        self.assertAlmostEqual(d1, d2)

    def test_saving_loading(self):
        # get the cwd so we can create a couple test files that we'll remove later
        dir = os.getcwd()
        self.identity_dq.save(dir + '/identity.json')
        # load it back in
        loaded_unit = DualQuaternion.from_file(dir + '/identity.json')
        self.assertEqual(self.identity_dq, loaded_unit)
        # clean up
        os.remove(dir + '/identity.json')

    def test_loading_illegal(self):
        self.assertRaises(IOError, DualQuaternion.from_file, 'boguspath')

    def test_sclerp_position(self):
        """test Screw Linear Interpolation for diff position, same orientation"""
        dq1 = DualQuaternion.from_translation_vector([2, 2, 2])
        dq2 = DualQuaternion.from_translation_vector([3, 4, -2])
        interpolated1 = DualQuaternion.sclerp(dq1, dq2, 0.5)
        expected1 = DualQuaternion.from_translation_vector([2.5, 3, 0])
        self.assertEqual(interpolated1, expected1)
        interpolated2 = DualQuaternion.sclerp(dq1, dq2, 0.1)
        expected2 = DualQuaternion.from_translation_vector([2.1, 2.2, 1.6])
        self.assertEqual(interpolated2, expected2)

    def test_sclerp_orientation(self):
        """test Screw Linear Interpolation for diff orientation, same position"""
        T_id = DualQuaternion.identity().homogeneous_matrix()
        T_id[0:2, 0:2] = np.array([[0, -1], [1, 0]])  # rotate 90 around z
        dq2 = DualQuaternion.from_homogeneous_matrix(T_id)
        interpolated1 = DualQuaternion.sclerp(self.identity_dq, dq2, 0.5)
        T_exp = DualQuaternion.identity().homogeneous_matrix()
        sq22 = np.sqrt(2)/2
        T_exp[0:2, 0:2] = np.array([[sq22, -sq22], [sq22, sq22]])  # rotate 45 around z
        expected1 = DualQuaternion.from_homogeneous_matrix(T_exp)
        self.assertEqual(interpolated1, expected1)
        interpolated2 = DualQuaternion.sclerp(self.identity_dq, dq2, 0)
        interpolated3 = DualQuaternion.sclerp(self.identity_dq, dq2, 1)
        self.assertEqual(interpolated2, self.identity_dq)
        self.assertEqual(interpolated3, dq2)

    def test_sclerp_screw(self):
        """Interpolating with ScLERP should yield same result as interpolating with screw parameters
        ScLERP is a screw motion interpolation with constant rotation and translation speeds. We can
        simply interpolate screw parameters theta and d and we should get the same result.
        """
        taus = [0., 0.23, 0.6, 1.0]
        l, m, theta, d = self.normalized_dq.screw()
        for tau in taus:
            # interpolate using sclerp
            interpolated_dq = DualQuaternion.sclerp(self.identity_dq, self.normalized_dq, tau)
            # interpolate using screw: l and m stay the same, theta and d vary with tau
            interpolated_dq_screw = DualQuaternion.from_screw(l, m, tau*theta, tau*d)
            self.assertEqual(interpolated_dq, interpolated_dq_screw)

    def test_exp_unit(self):
        """Validate exponential yields a unit dual quaternion"""
        pure_dq = DualQuaternion.from_dq_array([0, 1, 2, 3, 0, 2, -1, 1])
        exp = pure_dq.exp()
        self.assertTrue(exp.is_normalized)

    def test_exp_translation(self):
        """Exp mapping from a vector with translation component only results in a DQ with real part = 1"""
        translation = DualQuaternion.from_dq_array([0, 0, 0, 0, 0, 1, 2, 3])
        exp = translation.exp()
        self.assertEqual(exp.q_r, Quaternion(1,0,0,0))

    def test_log_pure(self):
        """Validate logarithm yields a pure dual quaternion"""
        log = self.normalized_dq.log()
        self.assertAlmostEqual(log.q_r.w, 0)
        self.assertAlmostEqual(log.q_d.w, 0)

        # twist of identity is 0
        self.assertEqual(DualQuaternion.identity().log(), DualQuaternion(Quaternion([0,0,0,0]), Quaternion([0,0,0,0])))

    def test_exp_log_identity(self):
        """
        Taking exp of the the log should yield original result
        
        Note: the inverse does not hold true in general (log(exp(dq)) != dq)
        """
        pure_dq = DualQuaternion.from_dq_array([0, 1, 2, 3, 0, 2, -1, 1]).normalized()
        self.assertEqual(pure_dq.log().exp(), pure_dq)

    def test_pow(self):
        expected_result = self.normalized_dq * self.normalized_dq
        received_result = self.normalized_dq.pow(2)
        print("Expected result:", expected_result)
        print("Received result:", received_result)
        print("nomrmalized_dq:", self.normalized_dq)
        self.assertEqual(received_result, expected_result)

    def test_sqrt_pow(self):
        """sqrt(dq) ?= pow(dq, 0.5)"""
        sqrt = DualQuaternion.sqrt(self.normalized_dq)
        pow12 = self.normalized_dq.pow(0.5)
        self.assertEqual(sqrt, pow12)

    def test_sclerp_shortest_path(self):
        """Test that ScLERP always chooses the shortest path between two dual quaternions
        
        Since dual quaternions have a double cover property (both dq and -dq represent
        the same transformation), ScLERP should automatically choose the shorter path
        by ensuring the quaternion dot product is positive.
        """
        # Test case 1: 200 degree rotation (should use 160° short path)
        start_dq = DualQuaternion.identity()
        
        angle_200 = 200 * np.pi / 180  # 200 degrees in radians
        end_rotation = Quaternion(axis=[0, 0, 1], angle=angle_200)
        end_translation = [1, 0, 0]
        end_dq = DualQuaternion.from_quat_pose_array([
            end_rotation.w, end_rotation.x, end_rotation.y, end_rotation.z,
            *end_translation
        ])
        
        # Test with the original dual quaternion vs negated version
        midpoint_original = DualQuaternion.sclerp(start_dq, end_dq, 0.5)
        end_dq_negated = DualQuaternion(-end_dq.q_r, -end_dq.q_d)
        midpoint_negated = DualQuaternion.sclerp(start_dq, end_dq_negated, 0.5)
        
        # Both should represent the same transformation at the endpoints
        self.assertEqual(end_dq, end_dq_negated)
        
        # The implementation should automatically choose the positive dot product path
        # So both interpolations should yield similar results (both taking the shorter path)
        _, _, theta_mid_original, _ = midpoint_original.screw()
        _, _, theta_mid_negated, _ = midpoint_negated.screw()
        
        self.assertAlmostEqual(abs(theta_mid_original), abs(theta_mid_negated), places=2)
        
        # The midpoint angle should be around 80° (half of 160° short path)
        # rather than 100° (half of 200° long path)
        expected_short_path_mid = (360 - 200) * np.pi / 180 / 2  # 80 degrees
        actual_mid = abs(theta_mid_original)
        
        # Should be closer to 80° than to 100°
        self.assertLess(abs(actual_mid - expected_short_path_mid), 
                       abs(actual_mid - angle_200/2))
        
        # Test case 2: 270 degree rotation (should use 90° short path)
        large_angle = 270 * np.pi / 180  # 270 degrees
        large_rotation = Quaternion(axis=[0, 1, 0], angle=large_angle)
        large_dq = DualQuaternion.from_quat_pose_array([
            large_rotation.w, large_rotation.x, large_rotation.y, large_rotation.z,
            0, 1, 0  # Translation along Y
        ])
        
        midpoint_large = DualQuaternion.sclerp(start_dq, large_dq, 0.5)
        midpoint_large_neg = DualQuaternion.sclerp(start_dq, 
                                                   DualQuaternion(-large_dq.q_r, -large_dq.q_d), 
                                                   0.5)
        
        # Both should take the same short path
        _, _, theta_large_mid, _ = midpoint_large.screw()
        _, _, theta_large_mid_neg, _ = midpoint_large_neg.screw()
        
        self.assertAlmostEqual(abs(theta_large_mid), abs(theta_large_mid_neg), places=2)
        
        # The midpoint should be around 45° (half of 90° short path)
        expected_short_mid = np.pi / 4  # 45 degrees
        self.assertLess(abs(theta_large_mid), np.pi/2)  # Should be less than 90°
        self.assertAlmostEqual(abs(theta_large_mid), expected_short_mid, places=1)
        
        # Test case 3: Rotation that exceeds 180° (181°, should use 179° short path)
        critical_angle = 181 * np.pi / 180  # 181 degrees
        critical_rotation = Quaternion(axis=[1, 0, 0], angle=critical_angle)
        critical_dq = DualQuaternion.from_quat_pose_array([
            critical_rotation.w, critical_rotation.x, critical_rotation.y, critical_rotation.z,
            0, 0, 1  # Translation along Z
        ])
        
        dot_critical = start_dq.q_r.w * critical_dq.q_r.w + np.dot(start_dq.q_r.vector, critical_dq.q_r.vector)
        
        # At 181°, the dot product should be negative (indicating long path)
        self.assertLess(dot_critical, 0)
        
        # ScLERP should automatically use the short path
        midpoint_critical = DualQuaternion.sclerp(start_dq, critical_dq, 0.5)
        _, _, theta_critical_mid, _ = midpoint_critical.screw()
        
        # Should be around 89.5° (half of 179° short path), not ~90.5° (half of 181°)
        expected_short_mid = (360 - 181) * np.pi / 180 / 2  # Half of the 179° short path
        self.assertLess(abs(theta_critical_mid), np.pi/2)  # Should be less than 90°
        self.assertAlmostEqual(abs(theta_critical_mid), expected_short_mid, places=1)


if __name__ == '__main__':
    unittest.main()
