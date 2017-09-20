class Material:
    def __init__(self, name, illum, ior, ka, kd, ks, absorption, scattering, asymmetry, scale):
        self.name = name
        self.illum = illum
        self.ior = ior
        self.absorption = np.array(absorption)
        self.scattering = np.array(scattering)
        self.g = np.array(asymmetry)
        self.scale = scale
        self.ka = np.array(ka)
        self.kd = np.array(kd)
        self.ks = np.array(ks)
        
    def __str__(self):
        s = "newmtl " + self.name + "\n"
        s += "Ka " + str(" ".join([str(a) for a in self.ka])) + "\n"
        s += "Kd " + str(" ".join([str(a) for a in self.kd])) + "\n"
        s += "Ks " + str(" ".join([str(a) for a in self.ks])) + "\n"
        s += "Ni " + str(self.ior) + "\n"
        s += "Sa " + str(" ".join([str(a) for a in self.absorption])) + "\n"
        s += "Ss " + str(" ".join([str(a) for a in self.scattering])) + "\n"
        s += "Sg " + str(" ".join([str(a) for a in self.g])) + "\n"
        s += "Sc " + str(self.scale) + "\n"
        s += "illum " + str(self.illum) + "\n"
        return s

apple = Material("apple", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.003000, 0.003400, 0.004600],scattering=[2.290000, 2.390000, 1.970000], asymmetry=[0.000000, 0.000000, 0.000000], scale=100.000000)
marble = Material("marble", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.002100, 0.004100, 0.007100],scattering=[2.190000, 2.620000, 3.000000], asymmetry=[0.000000, 0.000000, 0.000000], scale=100.000000)
potato = Material("potato", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.002400, 0.009000, 0.120000],scattering=[0.680000, 0.700000, 0.550000], asymmetry=[0.000000, 0.000000, 0.000000], scale=100.000000)
skin = Material("skin", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.032000, 0.170000, 0.480000],scattering=[0.740000, 0.880000, 1.010000], asymmetry=[0.000000, 0.000000, 0.000000], scale=100.000000)
chocolate_milk = Material("chocolate_milk", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.007000, 0.030000, 0.100000],scattering=[7.352000, 9.142000, 10.588000], asymmetry=[0.862000, 0.838000, 0.806000], scale=10.000000)
soy_milk = Material("soy_milk", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.000100, 0.000500, 0.003400],scattering=[2.433000, 2.714000, 4.563000], asymmetry=[0.873000, 0.858000, 0.832000], scale=100.000000)
white_grapefruit = Material("white_grapefruit", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.096000, 0.131000, 0.395000],scattering=[3.513000, 3.669000, 5.237000], asymmetry=[0.548000, 0.545000, 0.565000], scale=100.000000)
reduced_milk = Material("reduced_milk", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.000100, 0.000200, 0.000500],scattering=[10.748000, 12.209000, 13.931000], asymmetry=[0.819000, 0.797000, 0.746000], scale=100.000000)
ketchup = Material("ketchup", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.061000, 0.970000, 1.450000],scattering=[0.180000, 0.070000, 0.030000], asymmetry=[0.000000, 0.000000, 0.000000], scale=100.000000)
whole_milk = Material("whole_milk", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.001100, 0.002400, 0.014000],scattering=[2.550000, 3.210000, 3.770000], asymmetry=[0.000000, 0.000000, 0.000000], scale=100.000000)
chicken = Material("chicken", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.015000, 0.077000, 0.190000],scattering=[0.150000, 0.210000, 0.380000], asymmetry=[0.000000, 0.000000, 0.000000], scale=100.000000)
beer = Material("beer", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.144900, 0.314100, 0.728600],scattering=[0.003700, 0.006900, 0.007400], asymmetry=[0.917000, 0.956000, 0.982000], scale=100.000000)
coffee = Material("coffee", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.166900, 0.228700, 0.307800],scattering=[0.270700, 0.282800, 0.297000], asymmetry=[0.907000, 0.896000, 0.880000], scale=100.000000)
shampoo = Material("shampoo", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.178000, 0.328000, 0.439000],scattering=[8.111000, 9.919000, 10.575000], asymmetry=[0.907000, 0.882000, 0.874000], scale=100.000000)
mustard = Material("mustard", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.057000, 0.061000, 0.451000],scattering=[16.447001, 18.535999, 6.457000], asymmetry=[0.155000, 0.173000, 0.351000], scale=1.000000)
mixed_soap = Material("mixed_soap", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.003000, 0.005000, 0.013000],scattering=[3.923000, 4.018000, 4.351000], asymmetry=[0.330000, 0.322000, 0.316000], scale=100.000000)
glycerine_soap = Material("glycerine_soap", 12, ior=1.3, ka=[0,0,0], kd=[0,0,0], ks=[0,0,0], absorption=[0.001000, 0.001000, 0.002000],scattering=[0.201000, 0.202000, 0.221000], asymmetry=[0.955000, 0.949000, 0.943000], scale=100.000000)
