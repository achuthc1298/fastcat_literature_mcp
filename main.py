from mcp.server.fastmcp import FastMCP, Context
from tc_python import __version__
from tc_python import *
from dotenv import load_dotenv
import numpy as np
import os
from google import genai
from google.genai import types
from pydantic import BaseModel

# Initialize FastMCP server
mcp = FastMCP("thermocalc-agent")

# @mcp.tool()
# async def get_alloy_info(alloy_name: str, ctx: Context) -> str:
#     """Get alloy composition information."""
#     # Access resource through MCP's resource system
#     resource_uri = f"alloy://{alloy_name}"
#     try:
#         # This is the proper way to access MCP resources
#         composition = await ctx.read_resource(resource_uri)
#         await ctx.info(f"Looking up alloy: {alloy_name}")
#         return f"Alloy {alloy_name} composition: {composition}"
#     except Exception as e:
#         return f"Error accessing alloy data: {str(e)}"

@mcp.tool()
async def get_alloy_properties(alloy_name: str) -> dict[str, float]:
    """Get alloy composition information."""
    from llama_index.core.readers import SimpleDirectoryReader
    from llama_index.core import Settings
    # from llama_index.llms.openai import OpenAI
    # from llama_index.embeddings.openai import OpenAIEmbedding
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
    from llama_index.readers.file import PagedCSVReader
    from llama_index.vector_stores.faiss import FaissVectorStore
    from llama_index.core.ingestion import IngestionPipeline
    from llama_index.core import VectorStoreIndex
    import faiss
    
    import pandas as pd
    


    # Load environment variables from a .env file
    load_dotenv()

    # Set the OpenAI API key environment variable
    # os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ['TC25B_HOME'] = os.getenv('TC25B_HOME')
    os.environ['LSHOST']= os.getenv('LSHOST')
    llm = GoogleGenAI(
    model="gemini-2.5-flash",
   # uses GOOGLE_API_KEY env var by default
    )

    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        embed_batch_size=100,
    )

    Settings.embed_model = embed_model
    Settings.llm = llm

    file_path_comp = os.environ['ALLOY_COMPOSITION_CSV']
    data_comp = pd.read_csv(file_path_comp)

    fais_index = faiss.IndexFlatL2(768)
    vector_store = FaissVectorStore(faiss_index=fais_index)
    csv_reader = PagedCSVReader()

    reader_comp = SimpleDirectoryReader( 
    input_files=[file_path_comp],
    file_extractor= {".csv": csv_reader}
    )

    docs_comp = reader_comp.load_data()

    pipeline_comp = IngestionPipeline(
    vector_store=vector_store,
    documents=docs_comp
    )

    nodes_comp = pipeline_comp.run()
    vector_store_index_comp = VectorStoreIndex(nodes_comp)
    query_engine_comp = vector_store_index_comp.as_query_engine(similarity_top_k=2)
    response_comp = query_engine_comp.query(f"What is the composition of {alloy_name}? Give it in a python dictionary format. If there is more than one alloy return the string 'More than one alloy found, please be more specific'")
    s = response_comp.response
    if s == "More than one alloy found, please be more specific":
        raise ValueError("Please ask the user provide a more specific alloy name.")

    import re, json, ast
    # Step 1: Extract dictionary-like string using regex
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        dict_str = match.group(0)

        # Try JSON first (in case the producer ever emits proper JSON), fall back to Python literal
        try:
            result_dict = json.loads(dict_str)
        except json.JSONDecodeError:
            result_dict = ast.literal_eval(dict_str)

        # Step 3: Use the dictionary
        print(result_dict)

    # Convert all values to floats if possible
    numeric_dict = {}
    for k, v in result_dict.items():
        try:
            numeric_dict[k] = float(v)
        except (ValueError, TypeError):
            numeric_dict[k] = v  # Keep original if it can't be converted

    # Now filter for non-zero entries
    filtered_dict = {k: v for k, v in numeric_dict.items() if isinstance(v, float) and v > 0.0}

    # Sort descending
    sorted_filtered_dict = {k: v for k, v in sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True)}

    # # Assuming result_dict is your dictionary
    # filtered_dict = {k: v for k, v in result_dict.items() if v > 0.0}

    # sorted_filtered_dict = {k: v for k, v in sorted(filtered_dict.items(), key=lambda item: item[1], reverse=True)}

    # Now filtered_dict contains only non-zero entries
    print(sorted_filtered_dict)

    file_path_props = os.environ['ALLOY_PROPERTIES_CSV']
    # data_props = pd.read_csv(file_path_props)

    reader_props = SimpleDirectoryReader( 
        input_files=[file_path_props],
        file_extractor= {".csv": csv_reader}
        )

    docs_props = reader_props.load_data()
    pipeline_props = IngestionPipeline(
        vector_store=vector_store,
        documents=docs_props
    )

    nodes_props = pipeline_props.run()
    vector_store_index_props = VectorStoreIndex(nodes_props)
    query_engine_props = vector_store_index_props.as_query_engine(similarity_top_k=2)
    response_props_melt = query_engine_props.query(f"What is the melting point of {alloy_name} in K, return only numbers in your response")
    response_props_sph = query_engine_props.query(f"What is the specific heat capacity of {alloy_name}, return only numbers in your response")
    response_props_density = query_engine_props.query(f"What is the density of {alloy_name} in kg/m^3, return only numbers in your response")
    response_props_thermal_cond = query_engine_props.query(f"What is the thermal conductivity of {alloy_name} in W/m-K, return only numbers in your response")
    print(response_props_melt.response)
    from google import genai
    from google.genai import types
    client_gem = genai.Client()
    response_database = client_gem.models.generate_content(
        model="gemini-2.5-flash", contents=f"""
    TCFE14: Steels/Fe-Alloys
    TCNI12: Nickel Alloys
    TCAL9: Aluminum Alloys
    TCTI6: Titanium Alloys
    TCHEA7: High-Entropy Alloys
    PURE5: Pure Elements

    return only the name of the ThermoCalc database that is best for {s}
    do not write any other text
    """
    )
    print(response_database.text)



    # Define elements and their weight fractions
    element_weights = sorted_filtered_dict

    divisor = 100.0

    # Using dictionary comprehension to divide each value
    divided_dict = {key: value / divisor for key, value in element_weights.items()}
    element_weights = divided_dict
    print(element_weights)

    element_list = list(element_weights.keys())

    database_name = response_database.text

    melting_temp = float(response_props_melt.response)

    specific_heat = float(response_props_sph.response)

    density = float(response_props_density.response)

    thermal_conductivity = float(response_props_thermal_cond.response)

    # weighted_sum = 0.0

    # === Thermo-Calc Calculations ===
       
    

    # ----- Config -----
    tol = 5e-2  # tolerance for comparing to 0.0 and 1.0

    # Alloy composition
    # element_weights = {
    #     'Fe': 0.79, 'W': 0.061, 'Mo': 0.05, 'Cr': 0.041, 'V': 0.02,
    #     'C': 0.009, 'Si': 0.0033, 'Mn': 0.003, 'Ni': 0.002
    # }
    
    
    element_list = list(element_weights.keys())

    with TCPython() as start:
        start.set_cache_folder("cache")
        start.set_ges_version(6)

        calculation = (
            start.select_database_and_elements(database_name, element_list)
            .get_system()
            .with_property_diagram_calculation()
            .with_axis(
                CalculationAxis(ThermodynamicQuantity.temperature())
                .set_min(melting_temp - 600)
                .set_max(melting_temp + 600)
                .with_axis_type(Linear().set_min_nr_of_steps(50))
            ).
        set_condition(ThermodynamicQuantity.temperature(), 1000)
        )

        # Loop to add composition conditions
        for elem, wt in list(element_weights.items())[1:]:
            calculation.set_condition(f"W({elem})", wt)

        # Perform calculation
        property_diagram = calculation.calculate()
        property_diagram.set_phase_name_style(PhaseNameStyle.ALL)

        groups = property_diagram.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(),
            ThermodynamicQuantity.volume_fraction_of_a_phase("LIQUID")
        )

    # Variables for solidus and liquidus temperatures
    solidus_temp = None
    liquidus_temp = None

    for group in groups.values():
            y_list = group.y
            x_list = group.x
            

            # Find last index where y == 0
            last_zero_idx = max((i for i, val in enumerate(y_list) if val == 0), default=None)

            # Find first index after last_zero_idx where y == 1.0
            first_one_idx = None
            if last_zero_idx is not None:
                for i in range(last_zero_idx + 1, len(y_list)):
                    if y_list[i] == 1.0:
                        first_one_idx = i
                        break

            if last_zero_idx is not None:
                solidus_temp = x_list[last_zero_idx]
            else:
                print("No 0/solidus temp found")

            if first_one_idx is not None:
                liquidus_temp = x_list[first_one_idx]
            else:
                print("No 1.0/liquidus temp found after last 0")


        



    with TCPython() as start2:
        start2.set_cache_folder("cache")
        start2.set_ges_version(6)

        resistivity_calc = (
            start2.select_database_and_elements(database_name, element_list)
            .get_system()
            .with_property_diagram_calculation()
            .with_axis(
                CalculationAxis(ThermodynamicQuantity.temperature())
                .set_min(liquidus_temp - 5)   # a small range around liquidus
                .set_max(liquidus_temp + 5)
                .with_axis_type(Linear().set_min_nr_of_steps(20))
            ).
        set_condition(ThermodynamicQuantity.temperature(), liquidus_temp)
        )

        # Set composition conditions
        for elem, wt in list(element_weights.items())[1:]:
            resistivity_calc.set_condition(f"W({elem})", wt)

        

        # Perform calculation
        resistivity_diagram = resistivity_calc.calculate()

        # Extract electric resistivity data grouped by temperature
        resistivity_groups = resistivity_diagram.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(),
            ThermodynamicQuantity.electric_resistivity()
        )

    # Find resistivity at liquidus temperature within tolerance
    resistivity_at_liquidus = None
    for group in resistivity_groups.values():
        for T, rho in zip(group.x, group.y):
            if abs(T - liquidus_temp) <= tol:
                resistivity_at_liquidus = rho
                break
        if resistivity_at_liquidus is not None:
            break

    if resistivity_at_liquidus is None:
        raise RuntimeError("Could not find resistivity near the liquidus temperature.")

    print(f"Solidus Temperature (tol={tol}): {solidus_temp} K")
    print(f"Liquidus Temperature (tol={tol}): {liquidus_temp} K")
    print(f"Electric resistivity at liquidus ({liquidus_temp} K, tol={tol}): {resistivity_at_liquidus}")
    

    wavelength = 1070  # Given wavelength value

    

    # Calculate absorptive Drude value
    absorp_drude_calc = 0.365 * np.sqrt(resistivity_at_liquidus / (wavelength*1e-9))


    


    return {
    "Specific Heat Capacity in J/kg-K": specific_heat,
    "Laser Absorptivity": absorp_drude_calc,
    "Thermal Conductivity in W/m-K": thermal_conductivity,  
    "Density in kg/m^3": density,
    "Melting Temperature in K": melting_temp,
    "Solidus temperature in K": solidus_temp,
    "Liquidus_temperature in K": liquidus_temp
    }


@mcp.tool()
async def get_unknown_new_alloy_properties(unknown_new_alloy_composition: str) -> dict[str, float]:
    """Get alloy composition information."""
    # Load environment variables from a .env file
    load_dotenv()

    # Set the OpenAI API key environment variable
    # os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    os.environ['TC25B_HOME'] = os.getenv('TC25B_HOME')
    os.environ['LSHOST']= os.getenv('LSHOST')


    client_gem = genai.Client()

    

    response_alloy_comp = client_gem.models.generate_content(
        model="gemini-2.5-flash", contents=f"""
    {unknown_new_alloy_composition} is a new alloy.

    return only the python dictionary of the alloy composition, with elements as keys in double quotes and their weight fractions as decimals less then one as values.
    do not write any other text
    """
    )
    s=response_alloy_comp.text

    print(s)
    import re, json, ast
    # Step 1: Extract dictionary-like string using regex
    match = re.search(r'\{.*\}', s, re.DOTALL)
    if match:
        dict_str = match.group(0)

        # Try JSON first (in case the producer ever emits proper JSON), fall back to Python literal
        try:
            result_dict = json.loads(dict_str)
        except json.JSONDecodeError:
            result_dict = ast.literal_eval(dict_str)

        # Step 3: Use the dictionary
        print(result_dict)

        # Convert all values to floats if possible
    numeric_dict = {}
    for k, v in result_dict.items():
        try:
            numeric_dict[k] = float(v)
        except (ValueError, TypeError):
            numeric_dict[k] = v  # Keep original if it can't be converted
    print(numeric_dict)

    # Sort descending
    sorted_filtered_dict = {k: v for k, v in sorted(numeric_dict.items(), key=lambda item: item[1], reverse=True)}


    print(sorted_filtered_dict)

    response_database = client_gem.models.generate_content(
    model="gemini-2.5-flash", contents=f"""
    Guess the melting point for this alloy,
    {s}
    do not write any other text, only write numbers in response.
    """
    )
    mpt = float(response_database.text)
    print(mpt)

    response_database = client_gem.models.generate_content(
    model="gemini-2.5-flash", contents=f"""
    TCFE14: Steels/Fe-Alloys
    TCNI12: Nickel Alloys
    TCAL9: Aluminum Alloys
    TCTI6: Titanium Alloys
    TCHEA7: High-Entropy Alloys
    PURE5: Pure Elements

    return only the name of the ThermoCalc database that is best for {s}
    do not write any other text
    """
    )
    database_name = response_database.text
    print(database_name)

        # ----- Config -----
    tol = 5e-2  # tolerance for comparing to 0.0 and 1.0

    # Alloy composition
    element_weights = sorted_filtered_dict
    
    print(element_weights)
    
    element_list = list(element_weights.keys())

    #################################################################

    with TCPython() as start:
        start.set_cache_folder("cache")
        start.set_ges_version(6)

        calculation = (
            start.select_database_and_elements(database_name, element_list)
            .get_system()
            .with_property_diagram_calculation()
            .with_axis(
                CalculationAxis(ThermodynamicQuantity.temperature())
                .set_min(500)
                .set_max(3000)
                .with_axis_type(Linear().set_min_nr_of_steps(50))
            ).
        set_condition(ThermodynamicQuantity.temperature(), 1000)
        )

        # Loop to add composition conditions
        for elem, wt in list(element_weights.items())[1:]:
            calculation.set_condition(f"W({elem})", wt)

        # Perform calculation
        property_diagram = calculation.calculate()
        property_diagram.set_phase_name_style(PhaseNameStyle.ALL)

        groups = property_diagram.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(),
            ThermodynamicQuantity.volume_fraction_of_a_phase("LIQUID")
        )

    # Variables for solidus and liquidus temperatures
    solidus_temp = None
    liquidus_temp = None

    for group in groups.values():
            y_list = group.y
            x_list = group.x
            

            # Find last index where y == 0
            last_zero_idx = max((i for i, val in enumerate(y_list) if val == 0), default=None)

            # Find first index after last_zero_idx where y == 1.0
            first_one_idx = None
            if last_zero_idx is not None:
                for i in range(last_zero_idx + 1, len(y_list)):
                    if y_list[i] == 1.0:
                        first_one_idx = i
                        break

            if last_zero_idx is not None:
                solidus_temp = x_list[last_zero_idx]
            else:
                print("No 0/solidus temp found")

            if first_one_idx is not None:
                liquidus_temp = x_list[first_one_idx]
            else:
                print("No 1.0/liquidus temp found after last 0")


    #################################################################

    with TCPython() as start2:
        start2.set_cache_folder("cache")
        start2.set_ges_version(6)

        resistivity_calc = (
            start2.select_database_and_elements(database_name, element_list)
            .get_system()
            .with_property_diagram_calculation()
            .with_axis(
                CalculationAxis(ThermodynamicQuantity.temperature())
                .set_min(liquidus_temp - 5)   # a small range around liquidus
                .set_max(liquidus_temp + 5)
                .with_axis_type(Linear().set_min_nr_of_steps(20))
            ).
        set_condition(ThermodynamicQuantity.temperature(), liquidus_temp)
        )

        # Set composition conditions
        for elem, wt in list(element_weights.items())[1:]:
            resistivity_calc.set_condition(f"W({elem})", wt)

        

        # Perform calculation
        resistivity_diagram = resistivity_calc.calculate()

        # Extract electric resistivity data grouped by temperature
        resistivity_groups = resistivity_diagram.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(),
            ThermodynamicQuantity.electric_resistivity()
        )

    # Find resistivity at liquidus temperature within tolerance
    resistivity_at_liquidus = None
    for group in resistivity_groups.values():
        for T, rho in zip(group.x, group.y):
            if abs(T - liquidus_temp) <= tol:
                resistivity_at_liquidus = rho
                break
        if resistivity_at_liquidus is not None:
            break

    if resistivity_at_liquidus is None:
        raise RuntimeError("Could not find resistivity near the liquidus temperature.")

    print(f"Solidus Temperature (tol={tol}): {solidus_temp} K")
    print(f"Liquidus Temperature (tol={tol}): {liquidus_temp} K")
    print(f"Electric resistivity at liquidus ({liquidus_temp} K, tol={tol}): {resistivity_at_liquidus}")

    wavelength = 1070  # Given wavelength value

        
    # Calculate absorptive Drude value
    absorp_drude_calc = 0.365 * np.sqrt(resistivity_at_liquidus / (wavelength*1e-9))



    #################################################################

    with TCPython() as start3:
        start3.set_cache_folder("cache")
        start3.set_ges_version(6)

        thermal_cond_calc = (
            start3.select_database_and_elements(database_name, element_list)
            .get_system()
            .with_property_diagram_calculation()
            .with_axis(
                CalculationAxis(ThermodynamicQuantity.temperature())
                .set_min(298)   # a small range around room temperature
                .set_max(300)
                .with_axis_type(Linear().set_min_nr_of_steps(5))
            ).
        set_condition(ThermodynamicQuantity.temperature(), 298)
        )

        # Set composition conditions
        for elem, wt in list(element_weights.items())[1:]:
            thermal_cond_calc.set_condition(f"W({elem})", wt)

        

        # Perform calculation
        thermal_cond_diagram = thermal_cond_calc.calculate()

        # Extract thermal conductivity data grouped by temperature
        thermal_cond_groups = thermal_cond_diagram.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(),
            ThermodynamicQuantity.thermal_conductivity()
        )

    # Find thermal conductivity at room temperature within tolerance
    k_at_room_temp = None
    for group in thermal_cond_groups.values():
        for T, k in zip(group.x, group.y):
            if abs(T - 298) <= tol:
                k_at_room_temp = k
                break
        if k_at_room_temp is not None:
            break

    if k_at_room_temp is None:
        raise RuntimeError("Could not find thermal conductivity near room temperature.")



    #################################################################

    with TCPython() as start4:
        start4.set_cache_folder("cache")
        start4.set_ges_version(6)

        specific_heat_calc = (
            start4.select_database_and_elements(database_name, element_list)
            .get_system()
            .with_property_diagram_calculation()
            .with_axis(
                CalculationAxis(ThermodynamicQuantity.temperature())
                .set_min(298)   # a small range around room temp
                .set_max(300)
                .with_axis_type(Linear().set_min_nr_of_steps(5))
            ).
        set_condition(ThermodynamicQuantity.temperature(), 298)
        )

        # Set composition conditions
        for elem, wt in list(element_weights.items())[1:]:
            specific_heat_calc.set_condition(f"W({elem})", wt)

        # Perform calculation
        specific_heat_diagram = specific_heat_calc.calculate()

        # Extract specific heat capacity data grouped by temperature
        specific_heat_groups = specific_heat_diagram.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(),
            ThermodynamicQuantity.user_defined_function('HM.T')
        )

    # Find specific heat capacity at room temperature within tolerance
    specific_heat_at_room_temp = None
    for group in specific_heat_groups.values():
        for T, cp in zip(group.x, group.y):
            if abs(T - 298) <= tol:
                specific_heat_at_room_temp = cp
                break
        if specific_heat_at_room_temp is not None:
            break

    if specific_heat_at_room_temp is None:
        raise RuntimeError("Could not find specific heat capacity near the room temperature.")



    #################################################################
    with TCPython() as start5:
        start5.set_cache_folder("cache")
        start5.set_ges_version(6)

        mass_calc = (
            start5.select_database_and_elements(database_name, element_list)
            .get_system()
            .with_property_diagram_calculation()
            .with_axis(
                CalculationAxis(ThermodynamicQuantity.temperature())
                .set_min(298)   # a small range around room temp
                .set_max(300)
                .with_axis_type(Linear().set_min_nr_of_steps(5))
            ).
        set_condition(ThermodynamicQuantity.temperature(), 298)
        )

        # Set composition conditions
        for elem, wt in list(element_weights.items())[1:]:
            mass_calc.set_condition(f"W({elem})", wt)

        # Perform calculation
        mass_diagram = mass_calc.calculate()

        # Extract mass data grouped by temperature
        mass_groups = mass_diagram.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(),
            ThermodynamicQuantity.user_defined_function('B')
        )

    # Find mass at room temperature within tolerance
    mass_at_room_temp = None
    for group in mass_groups.values():
        for T, mass_val in zip(group.x, group.y):
            if abs(T - 298) <= tol:
                mass_at_room_temp = mass_val
                break
        if mass_at_room_temp is not None:
            break

    if mass_at_room_temp is None:
        raise RuntimeError("Could not find mass near the room temperature.")


    with TCPython() as start6:
        start6.set_cache_folder("cache")
        start6.set_ges_version(6)

        volume_calc = (
            start6.select_database_and_elements(database_name, element_list)
            .get_system()
            .with_property_diagram_calculation()
            .with_axis(
                CalculationAxis(ThermodynamicQuantity.temperature())
                .set_min(298)   # a small range around room temp
                .set_max(300)
                .with_axis_type(Linear().set_min_nr_of_steps(5))
            ).
        set_condition(ThermodynamicQuantity.temperature(), 298)
        )

        # Set composition conditions
        for elem, wt in list(element_weights.items())[1:]:
            volume_calc.set_condition(f"W({elem})", wt)

        # Perform calculation
        volume_diagram = volume_calc.calculate()

        # Extract volume data grouped by temperature
        volume_groups = volume_diagram.get_values_grouped_by_quantity_of(
            ThermodynamicQuantity.temperature(),
            ThermodynamicQuantity.user_defined_function('V')
        )

    # Find volume at room temperature within tolerance
    volume_at_room_temp = None
    for group in volume_groups.values():
        for T, vol_val in zip(group.x, group.y):
            if abs(T - 298) <= tol:
                volume_at_room_temp = vol_val
                break
        if volume_at_room_temp is not None:
            break

    if volume_at_room_temp is None:
        raise RuntimeError("Could not find volume near the room temperature.")

    density_at_room_temp = mass_at_room_temp / volume_at_room_temp
    melting_temp = (liquidus_temp+solidus_temp)/2.0

    return {
        "specific_heat_at_room_temp": specific_heat_at_room_temp,
        "absorp_drude_calc": absorp_drude_calc,
        "thermal_conductivity_at_room_temp": k_at_room_temp,
        "density_at_room_temp": density_at_room_temp,
        "melting_point": melting_temp,
        "temp_liquidus": liquidus_temp,
        "temp_solidus": solidus_temp
        }


    

    


@mcp.resource("resource://composition/{name}")
async def get_alloy_composition(name: str) -> str:
    """Key value of alloy composition"""
    compositions = {
        "ss316l": "Cr 0.1, Fe 0.9",
        "ti64": "Ti 0.90, Al 0.06, V 0.04",
    }
    if name not in compositions:
        raise ValueError(f"Unknown alloy: {name}")
    return compositions[name]

@mcp.resource("file://documents/{name}")
def get_document(name: str) -> str:
    """Read a document by name."""
    # This would normally read from disk
    return f"Content of {name}"


@mcp.resource("config://settings")
def get_settings() -> str:
    """Get application settings."""
    return """{
  "theme": "dark",
  "language": "en",
  "debug": false
}"""
@mcp.resource("resource://greeting")
def get_greeting() -> str:
    return "hello world."

@mcp.resource("printers://abbreviation")
async def get_printer() -> str:
    """Key value of printers"""
    printers = {
        "lpbf": "laser powder bed fusion",
        "sls": "selective laser sintering",
    }
    return printers
    # if name not in printers:
    #     raise ValueError(f"Unknown printer: {name}")
    # return printers[name]

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')
