pub const LOG_PAGE_FAULTS: bool = false;

// Advertising a hypervisor (CPUID.1:ECX[31] + the "VMwareVMware" vendor in leaf 0x40000000)
// is pure downside here: the VMware backdoor port (0x5658) itself is not implemented, so a
// port-based probe never confirms us anyway, while the CPUID-based anti-VM checks in
// era-appropriate copy protection (SecuROM 7, StarForce, TAGES) do — and those refuse to run
// rather than degrade.
pub const VMWARE_HYPERVISOR_PORT: bool = false;
