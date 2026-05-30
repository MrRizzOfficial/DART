library ieee;

package shared_params is
    CONSTANT DATA_WIDTH := 16;
    CONSTANT SSR_RATE := 8;

    type sample is record
        i : std_logic_vector(DATA_WIDTH - 1 downto 0);
        q : std_logic_vector(DATA_WIDTH - 1 downto 0);
    end record sample

    type ssr_array is array(0 to SSR_RATE - 1) of sample;

end package;

package body shared_params is
end package body;