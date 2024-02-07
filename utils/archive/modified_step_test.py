def step(self, mode='equal'):
        """
        Distribute dye to outgoing edges based on current dye concentration and incoming edges.
        Parameters:
        - mode (str): 'equal' or 'proportional'. 'equal' divides the dye equally among edges, 
                    while 'proportional' divides it based on edge velocities.
        """
        # Calculate total incoming dye
        incoming_dye_total = sum(edge.C_end for edge in self.incoming_edges)
        self.incoming_dye = incoming_dye_total

        #all dye gets redistributed, so prior concentration is 0, this is checking how much dye is collecting at node
        self.prior_concentration = incoming_dye_total

        # Calculate absorbed dye
        potential_absorption = self.incoming_dye * self.absorption_rate
        print(f"\t\t{self} potential absorption: {potential_absorption:.5f}")
        
        # Check if the absorption limit is reached
        if self.absorbed_amount == self.absorption_limit:
            absorbed_dye = 0                                            # no more absorption
            self.absorbed_amount = self.absorption_limit                # update absorbed amount to limit
            self.outgoing_dye = self.incoming_dye                       # all dye goes to outgoing edges

        # Check if the absorption limit will be reached    
        elif self.absorbed_amount + potential_absorption >= self.absorption_limit:
            absorbed_dye = self.absorption_limit - self.absorbed_amount # absorb up to limit
            self.absorbed_amount = self.absorption_limit                # update absorbed amount to limit
            self.outgoing_dye = self.incoming_dye - absorbed_dye        # remaining dye goes to outgoing edges

        # Otherwise, absorb all potential absorption
        else:
            absorbed_dye = potential_absorption
            self.absorbed_amount += absorbed_dye
            self.outgoing_dye = self.incoming_dye - absorbed_dye

        # self.outgoing_dye = self.incoming_dye - absorbed_dye
        # print(f"\t\t{self} absorbed dye: {absorbed_dye:.5f}")

        # Distribute total dye to outgoing edges
        if len(self.outgoing_edges) == 0:
            # If there are no outgoing edges, the dye stays in the node, node is a sink
            self.dye_concentration += self.incoming_dye - self.outgoing_dye
            return
        
        if mode == 'equal':
            equal_concentration = self.outgoing_dye / len(self.outgoing_edges)
            for edge in self.outgoing_edges:
                edge.start_dye_concentration = equal_concentration
            assert outgoing_value_check == self.outgoing_dye, f"Outgoing dye values do not sum to total outgoing dye value {self.outgoing_dye}"
            self.dye_concentration = 0

        elif mode == 'proportional':
            total_velocity = sum(edge.u for edge in self.outgoing_edges)
            outgoing_value_check = 0
            for edge in self.outgoing_edges:
                edge.start_dye_concentration = self.outgoing_dye * (edge.u / total_velocity)
                outgoing_value_check += edge.start_dye_concentration
            assert outgoing_value_check == self.outgoing_dye, f"Outgoing dye values do not sum to total outgoing dye value {self.outgoing_dye}"
            self.dye_concentration = 0

        else:
            raise ValueError("Mode should be either 'equal' or 'proportional'")
        
        # This is to avoid negative dye concentrations from rounding errors
        if self.dye_concentration < 1e-10:  
            self.dye_concentration = 0
        
        assert self.dye_concentration >= 0, f"Node {self} has negative dye concentration {self.dye_concentration}"
        
def step(self, mode='equal'):
    """
    Distribute dye to outgoing edges based on current dye concentration and incoming edges.
    Parameters:
    - mode (str): 'equal' or 'proportional'. 'equal' divides the dye equally among edges, 
                while 'proportional' divides it based on edge velocities.
    """
    # Calculate total incoming dye
    incoming_dye = sum(edge.C_end for edge in self.incoming_edges)
    self.incoming_dye = incoming_dye
    #print(f"\t\t{self} incoming dye: {incoming_dye:.5f}")

    # Caclulate absorbed dye
    potential_absorption = incoming_dye * self.absorption_rate
    
    # Check if the absorption limit is reached
    if self.absorption_limit != np.inf:   
        if self.absorbed_amount + potential_absorption > self.absorption_limit:
            absorbed_dye = self.absorption_limit - self.absorbed_amount
            self.absorbed_amount = self.absorption_limit  # update absorbed amount to limit
        else:
            absorbed_dye = potential_absorption
            self.absorbed_amount += absorbed_dye
    else:
        absorbed_dye = potential_absorption
        self.absorbed_amount += absorbed_dye

    incoming_dye -= absorbed_dye
    self.outgoing_dye = incoming_dye
    #print(f"\t\t{self} absorbed dye: {absorbed_dye:.5f}")
    
    # Calculate total dye
    total_dye = self.dye_concentration + incoming_dye
    #print(f"\t\t{self} total dye in node: {total_dye:.5f}, previous amount of dye: {self.dye_concentration:.5f}")

    #save prior concentration for references
    self.prior_concentration = total_dye

    # Distribute total dye to outgoing edges
    if not self.outgoing_edges:
        # If there are no outgoing edges, the dye stays in the node
        self.dye_concentration = total_dye
        return
    
    if mode == 'equal':
        equal_concentration = total_dye / len(self.outgoing_edges)
        for edge in self.outgoing_edges:
            edge.start_dye_concentration = equal_concentration
            
    elif mode == 'proportional':
        total_velocity = sum(edge.u for edge in self.outgoing_edges)
        for edge in self.outgoing_edges:
            edge.start_dye_concentration = total_dye * (edge.u / total_velocity)
            
    else:
        raise ValueError("Mode should be either 'equal' or 'proportional'")
    
    # Update node's dye concentration to account for distributed dye
    self.dye_concentration = total_dye - sum(edge.start_dye_concentration for edge in self.outgoing_edges)
    
    # This is to avoid negative dye concentrations from rounding errors
    if self.dye_concentration < 1e-10:  
        self.dye_concentration = 0
    
    assert self.dye_concentration >= 0, f"Node {self} has negative dye concentration {self.dye_concentration}"